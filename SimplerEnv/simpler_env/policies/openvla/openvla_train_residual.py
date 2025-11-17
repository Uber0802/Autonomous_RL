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
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)

import torch
from torch import nn
import numpy as np

def layer_init(layer, bias_const=0.0):
    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResidualPolicyNet(nn.Module):
    def __init__(self, instr_dim, hidden_dim, action_dim, action_high, action_low):
        super().__init__()
        self.action_dim = action_dim
        self.cont_dim = action_dim - 1
        input_dim = hidden_dim + action_dim + instr_dim

        self.fc = nn.Sequential(
            layer_init(nn.Linear(input_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512, self.cont_dim)
        # self.fc_logit = nn.Linear(512, 1)

        nn.init.zeros_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        # nn.init.zeros_(self.fc_logit.weight)
        # nn.init.zeros_(self.fc_logit.bias)

        action_range = action_high[:, :self.cont_dim] - action_low[:, :self.cont_dim]
        init_std = (action_range / 10.0).clamp(min=1e-3)
        self.log_std = nn.Parameter(torch.log(init_std.squeeze(0)))

    def forward(self, instr, hidden, a_base):
        hx = torch.cat([instr, hidden, a_base], dim=-1)
        hh = self.fc(hx)
        mu = self.fc_mu(hh)
        mu = torch.tanh(mu) * 0.1
        std = self.log_std.exp().clamp(1e-4, 0.2).expand_as(mu)

        return mu, std







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
        
        self.vla.eval() 
        self.unnorm_state = self.vla.get_action_stats(self.args.vla_unnorm_key)
        self.action_low = torch.tensor(self.unnorm_state["q01"], device=self.tpdv["device"]).unsqueeze(0)
        self.action_high = torch.tensor(self.unnorm_state["q99"], device=self.tpdv["device"]).unsqueeze(0)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.residual_policy = ResidualPolicyNet(384, 4096, 7, self.action_high, self.action_low).to(self.tpdv["device"])
        self.value_head_res = nn.Sequential(
            nn.Linear(384 + 4096 + 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        ).to(self.tpdv["device"])

        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        ).to(self.tpdv["device"])

        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],  
            lora_dropout=0.05,
            bias="none",
        )
        self.text_encoder = get_peft_model(self.text_encoder, lora_cfg)


        self.residual_optimizer = AdamW(list(self.residual_policy.parameters()) + list(self.value_head_res.parameters()) + list(self.text_encoder.parameters()), lr=7e-5)

        

    def encode_instruction(self, text):
        tokens = self.text_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.tpdv["device"])
        with torch.no_grad():
            output = self.text_encoder(**tokens)
            emb = output.last_hidden_state.mean(dim=1)  
        return emb


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

    # @torch.no_grad()
    # def get_action(self, x: dict, deterministic=False):
    #     temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
    #     do_sample = (temperature != 0.0)

    #     # VLA outputs
    #     features = self._preprocess_obs(x)
    #     values, action_vla, logprobs = self.vla.predict_action_batch(
    #         **features,
    #         unnorm_key=self.args.vla_unnorm_key,
    #         do_sample=do_sample,
    #         temperature=temperature,
    #     )

    #     hidden = self.vla.get_hidden(**features).to(torch.float32)
        
    #     if hidden.ndim > 2:
    #         hidden = hidden.mean(dim=1)

    #     decoded_norm = self.action_tokenizer.decode_token_ids_to_actions(action_vla.detach().cpu().numpy())
    #     actions = 0.5 * (decoded_norm + 1.0) * (self.action_high.cpu().numpy() - self.action_low.cpu().numpy()) + self.action_low.cpu().numpy()
    #     processed_actions = torch.from_numpy(actions).to(dtype=torch.float64, device=self.tpdv["device"])

    #     # Residual correction
    #     a_res = self.residual_policy(hidden)
    #     action_final = processed_actions + a_res
    #     normalized_action_final = 2 * (action_final - self.action_low) / (self.action_high - self.action_low) - 1.0
    #     token_ids_np = encode_actions_from_norm_openvla(normalized_action_final, self.action_tokenizer) 
    #     token_ids = torch.tensor(token_ids_np, device=self.tpdv["device"])

    #     return values.to(torch.float32), action_final.to(torch.float32), logprobs
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


    def get_action_temp(self, x: dict, do_sample, temperature, num_beams) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x)

        _, action, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
        )

        assert len(action.shape) == 2
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return action, logprobs

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

    # def evaluate_residual(self, hidden, a_base, a_final):
    #     mu, std = self.residual_policy(hidden, a_base)
    #     dist = torch.distributions.Normal(mu, std)
    #     a_res = a_final - a_base
    #     logprob = dist.log_prob(a_res)
    #     entropy = dist.entropy()
    #     values = self.value_head_res(torch.cat([hidden, a_base], dim=-1))
    #     return logprob, entropy, values

    def evaluate_residual(self, hidden, a_base, a_final, instr_emb):
        mu, std = self.residual_policy(instr_emb, hidden, a_base)

        # split
        a_res_cont = a_final[..., :-1] - a_base[..., :-1]

        # continuous part
        dist_cont = torch.distributions.Normal(mu, std)
        logprob_cont = dist_cont.log_prob(a_res_cont).sum(dim=-1, keepdim=True)
        entropy_cont = dist_cont.entropy().sum(dim=-1, keepdim=True)

        # discrete part
        # dist_disc = torch.distributions.Bernoulli(logits=logit)
        # logprob_disc = dist_disc.log_prob(a_res_disc).sum(dim=-1, keepdim=True)
        # entropy_disc = dist_disc.entropy().sum(dim=-1, keepdim=True)

        # combine
        logprob = logprob_cont 
        entropy = entropy_cont 

        # value head uses both continuous + discrete inputs
        values = self.value_head_res(torch.cat([instr_emb, hidden, a_base], dim=-1))
        return logprob, entropy, values


    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.eval()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        self.vla.save_pretrained(str(path))
        training_state = {
            "vh": self.vla.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        torch.save(training_state, path / "training_state.pt")
        torch.save(self.residual_policy.state_dict(), path / "residual.pt")

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
        self.residual_policy.load_state_dict(torch.load(path / "residual.pt"))

class OpenVLAPPORes:
    def __init__(self, all_args, policy: OpenVLAPolicy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = 0.2
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv
        self.tpdv_vn = self.policy.tpdv_vn

    def train_res_ppo_step(self, idx, total, batch):
        
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages, reward, hidden, base_action, action_untok = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)  # uint8
        actions = torch.tensor(actions).to(self.tpdv["device"])  # int32
        value_preds = torch.tensor(value_preds).to(**self.tpdv)
        returns = torch.tensor(returns).to(**self.tpdv_vn)  # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)
        returns_norm = returns.to(**self.tpdv)
        hidden = torch.tensor(hidden).to(**self.tpdv)
        base_action = torch.tensor(base_action).to(self.tpdv["device"])
        action_untok = torch.tensor(action_untok).to(self.tpdv["device"])
        instr_emb = self.policy.encode_instruction(instruct)


        # Policy loss
        logprob, entropy, values = self.policy.evaluate_residual(hidden, base_action, action_untok, instr_emb)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()


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

        #if abs(policy_loss.item()) < 5*abs(value_loss.item()):
        #    policy_loss = 5*policy_loss

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        # loss.backward()

        # if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
        #     grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla + self.policy.params_vh, self.ppo_grad_norm)
        #     self.policy.vh_optimizer.step()
        #     self.policy.vla_optimizer.step()
        #     self.policy.vh_optimizer.zero_grad()
        #     self.policy.vla_optimizer.zero_grad()
        # else:
        #     grad_norm = None
        loss.backward()
        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.residual_policy.parameters(), self.ppo_grad_norm)
            self.policy.residual_optimizer.step()
            self.policy.residual_optimizer.zero_grad()
        else:
            grad_norm = None
        # Logging to log.txt
        log_str = (
            f"[PPO Step {idx}/{total}] "
            f"Returns: {returns.mean().item():.4f} | "
            f"Advantages: {advantages.mean().item():.4f} | "
            f"Policy Loss: {policy_loss.item():.4f} | "
            f"Value Loss: {value_loss.item():.4f} | "
            f"Entropy Loss: {entropy_loss.item():.4f} | "
            f"Loss: {loss.item():.4f}\n"
        )

        with open(self.args.log, "a") as f:
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


    def train_res_ppo(self, buffer):
        print("train residual")
        train_info = defaultdict(lambda: [])

        # buffer
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for i in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_res_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info

    