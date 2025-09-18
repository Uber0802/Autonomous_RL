import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, BatchFeature
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class OpenVLAPolicy:
    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.tpdv = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.bfloat16)
        self.tpdv_vn = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.float32)
        self.action_scale = 1.0

        # openvla: register
        self.image_processor = PrismaticImageProcessor.from_pretrained(self.args.vla_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.vla_path, trust_remote_code=True, padding_side="left")
        self.processor = PrismaticProcessor.from_pretrained(
            self.args.vla_path,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )
        # self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )

        # openvla: lora
        if not self.args.vla_load_path:
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=min(self.args.vla_lora_rank, 16),
                lora_dropout=0.0,
                target_modules=[
                    "proj", "qkv", "fc1", "fc2",  # vision
                    "q", "kv", "fc3",  # project
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",  # llm
                ],
                init_lora_weights="gaussian"
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            self.vla = PeftModel.from_pretrained(self.vla, self.args.vla_load_path, is_trainable=True)
            print(f"VLA load: {self.args.vla_load_path}")

            if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
                path = Path(self.args.vla_load_path) / "dataset_statistics.json"
                ds = json.load(open(path, "r"))
                self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        # set value head trainable
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        self.vla.print_trainable_parameters()

        # openvla: optimizer
        self.params_vh = None
        self.params_vla = None
        self.vh_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        if self.args.vla_load_path:
            training_state_path = Path(self.args.vla_load_path) / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

                if "vh" in training_state:
                    self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
                else:
                    print("Warning: value_head state not found in training_state")

                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
                self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

                print(f"Optimizer load: {self.args.vla_load_path}")
            else:
                print(f"Warning: training_state not found in {training_state_path}")

    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        self.params_vla = [p for n, p in self.vla.named_parameters() if "value_head" not in n and p.requires_grad]
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)

    def _preprocess_obs(self, x: dict, action: torch.Tensor = None) -> BatchFeature:
        images = x["image"]
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 4
        assert images.shape[3] == 3
        assert images.dtype == torch.uint8

        assert isinstance(task_description, list)
        assert isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)

        images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
        images = images.to(**self.tpdv)

        torch.cuda.synchronize()

        # prompt
        if action is None:
            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: "
                           for t in task_description]
        else:
            assert isinstance(action, torch.Tensor)
            # action = action.cpu().numpy() # [B, dim]
            action_str = self.tokenizer.batch_decode(action)

            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: {a}</s>"
                           for t, a in zip(task_description, action_str)]
  
        inputs = self.processor(task_prompt, images, padding=True)
        inputs = inputs.to(**self.tpdv)
        torch.cuda.synchronize()

        if action is not None:
            inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    def get_action(self, x: dict, deterministic) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        do_sample = (temperature != 0.0)
        features = self._preprocess_obs(x)


        torch.cuda.synchronize()
        values, action, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
        )

        assert len(values.shape) == 2 and values.shape[1] == 1
        assert len(action.shape) == 2 and action.shape[0] == values.shape[0]
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return values, action, logprobs

    def get_action_temp(self, x: dict, do_sample, temperature, num_beams, num_return_sequences) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x)

        values, actions, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
            top_k=20,
            top_p=0.95,
            num_return_sequences= num_return_sequences,
        )

        assert len(actions.shape) == 2
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return values, actions, logprobs

    def get_value(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        value = self.vla.get_value(**features)

        assert len(value.shape) == 2 and value.shape[1] == 1

        return value

    def get_hidden(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        hs = self.vla.get_hidden(**features)

        return hs

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x, action)

        logprobs, entropy, values = self.vla.evaluate_action(
            **features,
            unnorm_key=self.args.vla_unnorm_key
        )

        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1
        assert len(entropy.shape) == 2 and entropy.shape[1] == 1
        assert len(values.shape) == 2 and values.shape[1] == 1

        return logprobs, entropy, values

    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.train()
        # self.vla.eval()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        self.vla.save_pretrained(str(path))
        training_state = {
            "vh": self.vla.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        torch.save(training_state, path / "training_state.pt")

        json.dump(self.vla.base_model.norm_stats, open(path / "dataset_statistics.json", "w"))

    def load(self, path: Path):
        del self.vla
        torch.cuda.empty_cache()

        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )
        self.vla = PeftModel.from_pretrained(self.vla, path, is_trainable=True)
        self.vla.print_trainable_parameters()

        if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
            ds = json.load(open(path / "dataset_statistics.json", "r"))
            self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        training_state_path = path / "training_state.pt"
        training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
        else:
            print("Warning: value_head state not found in training_state")

        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
        self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

class OpenVLAPPO:
    def __init__(self, all_args, policy: OpenVLAPolicy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = 0.2
        self.ppo_grad_norm = 5.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 5.0
        self.tpdv = self.policy.tpdv
        self.tpdv_vn = self.policy.tpdv_vn

    def train_ppo_step(self, idx, total, batch, episode):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)  # uint8
        actions = torch.tensor(actions).to(self.tpdv["device"])  # int32
        value_preds = torch.tensor(value_preds).to(**self.tpdv)
        returns = torch.tensor(returns).to(**self.tpdv_vn)  # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)
        returns_norm = returns.to(**self.tpdv)

        # Policy loss
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        if episode <= 10:
            self.ppo_clip = 1
        elif episode <= 15:
            self.ppo_clip = 0.5
        elif episode <= 20:
            self.ppo_clip = 0.3
        else:
            self.ppo_clip = 0.2
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        # policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()
        # policy_loss = -torch.min(surr1, surr2).mean()
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # if episode <= 10:
        #     logprob_b = logprob.float().sum(dim=-1, keepdim=True)  
        #     adv = advantages.float()
        #     adv = (adv - adv.mean()) / (adv.std() + 1e-8) 
        #     policy_loss = -(logprob_b * adv).mean()
        # else:
        #     policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()


        # print("episode :", episode, "idx :", idx, "ratio :", ratio)
        


        # Value loss
        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_clip_indicator = (value_pred_clipped - value_preds).abs() > self.ppo_clip
        value_clip_ratio = value_clip_indicator.to(**self.tpdv).mean()

        value_loss = value_loss.mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        # print("here")
        loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla + self.policy.params_vh, self.ppo_grad_norm)
            self.policy.vh_optimizer.step()
            self.policy.vla_optimizer.step()
            self.policy.vh_optimizer.zero_grad()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None
        # Logging to log.txt
        log_str = (
            f"[PPO Step {idx}/{total}] "
            f"Returns: {returns.mean().item():.4f} | "
            f"Advantages: {advantages.mean().item():.4f} | "
            f"Policy : {policy_loss.item():.4f} | "
            f"Value : {value_loss.item():.4f} | "
            f"Entropy : {entropy_loss.item():.4f}\n"
            f"loss : {loss.item():.4f}\n"
        )

        with open("./-1_sft_multitask_0.05_guide15.txt", "a") as f:
            f.write(log_str)



        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),

            value_clip_ratio=value_clip_ratio.item(),
            value_old_mean=value_preds.mean().item(),
            values_mean=values.mean().item(),
            returns_mean=returns.mean().item(),
            returns_norm_mean=returns_norm.mean().item(),
            logprob_mean=logprob.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_ppo_step_batch(self, idx, total, batch, episode):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs_image   = torch.as_tensor(obs_image)                
        actions     = torch.as_tensor(actions)                  
        value_preds = torch.as_tensor(value_preds, dtype=self.tpdv["dtype"])
        returns     = torch.as_tensor(returns,     dtype=self.tpdv_vn["dtype"])
        old_logprob = torch.as_tensor(old_logprob, dtype=self.tpdv["dtype"])
        advantages  = torch.as_tensor(advantages,  dtype=self.tpdv["dtype"])

        B  = obs_image.shape[0]
        mb = 8                    
        num_mb = (B + mb - 1) // mb

        self.policy.vh_optimizer.zero_grad(set_to_none=True)
        self.policy.vla_optimizer.zero_grad(set_to_none=True)

        policy_loss_sum = value_loss_sum = entropy_sum = 0.0
        ratio_mean_sum = ratio2_mean_sum = 0.0
        value_clip_ratio_sum = value_preds_sum = values_sum = returns_sum = 0.0
        logprob_sum = old_logprob_sum = 0.0
        n_seen = 0

        if episode <= 10:   clip = 0.5
        elif episode <= 15: clip = 0.3
        else:               clip = 0.2
        self.ppo_clip = clip

        for s in range(0, B, mb):
            e = min(s + mb, B); w = e - s

            obs = dict(
                image=obs_image[s:e].to(self.tpdv["device"], non_blocking=True),
                task_description=instruct[s:e],
            )
            actions_mb     = actions[s:e].to(self.tpdv["device"], non_blocking=True)
            value_preds_mb = value_preds[s:e].to(**self.tpdv)
            returns_mb     = returns[s:e].to(**self.tpdv_vn)
            old_logprob_mb = old_logprob[s:e].to(**self.tpdv)
            advantages_mb  = advantages[s:e].to(**self.tpdv)
            returns_norm_mb= returns_mb.to(**self.tpdv)

            logprob, entropy, values = self.policy.evaluate_actions(obs, actions_mb)
            ratio = torch.exp(logprob - old_logprob_mb)

            surr1 = ratio * advantages_mb
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            # print("episode :", episode, "idx :", idx, "ratio :", ratio)


            values_f32      = values.float()
            value_preds_f32 = value_preds_mb.float()
            value_pred_clipped = value_preds_f32 + (values_f32 - value_preds_f32).clamp(-clip, clip)
            err_c = returns_mb - value_pred_clipped
            err_o = returns_mb - values_f32
            value_loss = torch.max(huber_loss(err_o, self.ppo_huber_delta),
                                huber_loss(err_c, self.ppo_huber_delta)).mean()

            value_clip_ratio_mb = ((value_pred_clipped - value_preds_f32).abs() > clip).float().mean()
            entropy_loss = entropy.mean()

            loss_mb = policy_loss + 1 * value_loss - self.ppo_entropy_coef * entropy_loss
            loss_mb = loss_mb / num_mb
            loss_mb.backward()   

            n_seen += w
            policy_loss_sum += policy_loss.item() * w
            value_loss_sum  += value_loss.item()  * w
            entropy_sum     += entropy_loss.item() * w
            ratio_mean_sum  += ratio.mean().item() * w
            ratio2_mean_sum += (logprob - old_logprob_mb).mean().exp().item() * w
            value_clip_ratio_sum += value_clip_ratio_mb.item() * w
            value_preds_sum += value_preds_mb.mean().item() * w
            values_sum      += values.mean().item() * w
            returns_sum     += returns_mb.mean().item() * w
            logprob_sum     += logprob.mean().item() * w
            old_logprob_sum += old_logprob_mb.mean().item() * w

        grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla + self.policy.params_vh, self.ppo_grad_norm)
        self.policy.vh_optimizer.step(); self.policy.vla_optimizer.step()

        policy_loss_mean = policy_loss_sum / n_seen
        value_loss_mean  = value_loss_sum  / n_seen
        entropy_mean     = entropy_sum     / n_seen
        ratio_mean       = ratio_mean_sum  / n_seen
        ratio2_mean      = ratio2_mean_sum / n_seen
        value_clip_ratio = value_clip_ratio_sum / n_seen
        value_old_mean   = value_preds_sum / n_seen
        values_mean      = values_sum      / n_seen
        returns_mean     = returns_sum     / n_seen
        logprob_mean     = logprob_sum     / n_seen
        logprob_old_mean = old_logprob_sum / n_seen
        loss_for_log     = policy_loss_mean + 0.01 * value_loss_mean - self.ppo_entropy_coef * entropy_mean

        with open("./log.txt", "a") as f:
            f.write(f"[PPO Step {idx}/{total}] Returns: {returns_mean:.4f} | "
                    f"Advantages: {advantages.mean().item():.4f} | "
                    f"Policy Loss: {policy_loss_mean:.4f} | "
                    f"Value Loss: {value_loss_mean:.4f} | "
                    f"Entropy Loss: {entropy_mean:.4f}\n")

        info = dict(
            loss=float(loss_for_log),
            policy_loss=policy_loss_mean,
            value_loss=value_loss_mean,
            entropy_loss=entropy_mean,
            ratio=ratio_mean,
            ratio_median=ratio_mean,
            ratio_2=ratio2_mean,
            value_clip_ratio=value_clip_ratio,
            value_old_mean=value_old_mean,
            values_mean=values_mean,
            returns_mean=returns_mean,
            returns_norm_mean=returns_mean,
            logprob_mean=logprob_mean,
            logprob_old_mean=logprob_old_mean,
            grad_norm=grad_norm.item(),
        )
        return info



    def train_grpo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)  # uint8
        actions = torch.tensor(actions).to(self.tpdv["device"])  # int32
        # value_preds = torch.tensor(value_preds).to(**self.tpdv)
        # returns = torch.tensor(returns).to(**self.tpdv_vn) # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)

        # Policy loss
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = policy_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla, self.ppo_grad_norm)
            self.policy.vla_optimizer.step()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),

            logprob_mean=logprob.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_ppo(self, buffer, episode):
        train_info = defaultdict(lambda: [])

        # buffer
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                if idx >= 640:
                    break
                info = self.train_ppo_step(idx, minibatch_count, batch, episode)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info

    def train_grpo(self, buffer):
        train_info = defaultdict(lambda: [])

        # buffer
        buffer.compute_returns_grpo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_grpo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info
