"""
UniVLA LAM Policy — generates ACT tokens, decodes via LatentActionDecoder.

Key differences from OpenVLAPolicy:
  - predict_action_batch returns 4-tuple: (values, continuous_actions, act_token_ids, logprobs)
  - Replay buffer stores 32 ACT token IDs (not 7 bin tokens)
  - evaluate_actions takes ACT token IDs and reconstructs the teacher-forced input
  - Environment receives continuous actions directly (no bin-to-continuous conversion)
"""
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
from prismatic.extern.hf.modeling_prismatic import UniVLAForActionPredictionWithValueHead
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class UniVLALAMPolicy:
    """Policy wrapper for UniVLA with LAM decoder."""

    # ACT token constants (must match modeling_prismatic.py)
    NUM_ACT_TOKENS = 32
    ACT_TOKEN_START = 32001

    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.tpdv = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.bfloat16)
        self.tpdv_vn = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.float32)
        self.action_scale = 1.0

        # Load processor/tokenizer
        self.image_processor = PrismaticImageProcessor.from_pretrained(self.args.vla_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.vla_path, trust_remote_code=True, padding_side="left")
        self.processor = PrismaticProcessor.from_pretrained(
            self.args.vla_path,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )

        # Load model
        self.vla = UniVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )

        # Load action decoder
        action_decoder_path = getattr(self.args, 'action_decoder_path', '')
        if not action_decoder_path:
            action_decoder_path = str(Path(self.args.vla_path) / "action_decoder.pt")
        self.vla.load_action_decoder(action_decoder_path)

        # LoRA
        if not self.args.vla_load_path:
            # Temporarily detach action_decoder to avoid LoRA targeting its modules
            # (action_decoder has "proj", "q", "kv" which collide with vision/LLM targets)
            action_decoder = self.vla.action_decoder
            self.vla.action_decoder = None

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

            # Re-attach frozen action decoder
            self.vla.base_model.model.action_decoder = action_decoder
        else:
            self.vla = PeftModel.from_pretrained(self.vla, self.args.vla_load_path, is_trainable=True)
            print(f"VLA load: {self.args.vla_load_path}")

            if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
                path = Path(self.args.vla_load_path) / "dataset_statistics.json"
                ds = json.load(open(path, "r"))
                self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        # Value head trainable
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        self.vla.print_trainable_parameters()

        # Optimizers
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
                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
                self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])
                print(f"Optimizer load: {self.args.vla_load_path}")

    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        self.params_vla = [p for n, p in self.vla.named_parameters() if "value_head" not in n and p.requires_grad]
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)

    def _preprocess_obs(self, x: dict, action: torch.Tensor = None) -> BatchFeature:
        """Preprocess observations. When action is provided, it's ACT token IDs [B, 32]."""
        images = x["image"]
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor) and images.dtype == torch.uint8
        assert images.shape[3] == 3

        images = images.permute(0, 3, 1, 2).to(**self.tpdv)
        torch.cuda.synchronize()

        if action is None:
            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: "
                           for t in task_description]
        else:
            # action is [B, 32] ACT token IDs
            assert isinstance(action, torch.Tensor)
            action_str = self.tokenizer.batch_decode(action)
            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: {a}</s>"
                           for t, a in zip(task_description, action_str)]

        inputs = self.processor(task_prompt, images, padding=True)
        inputs = inputs.to(**self.tpdv)
        torch.cuda.synchronize()

        if action is not None:
            inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    def get_action(self, x: dict, deterministic) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (values [B,1], continuous_actions [B,7], act_token_ids [B,32], logprobs [B,1])

        continuous_actions: sent to environment
        act_token_ids: stored in replay buffer for evaluate_actions
        """
        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        do_sample = (temperature != 0.0)
        features = self._preprocess_obs(x)

        torch.cuda.synchronize()
        values, continuous_actions, act_token_ids, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
        )

        assert values.shape == (x["image"].shape[0], 1)
        assert continuous_actions.shape == (x["image"].shape[0], 7)
        assert act_token_ids.shape == (x["image"].shape[0], self.NUM_ACT_TOKENS)
        assert logprobs.shape == (x["image"].shape[0], 1)

        return values, continuous_actions, act_token_ids, logprobs

    def get_value(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)
        value = self.vla.get_value(**features)
        assert value.shape[1] == 1
        return value

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """action is [B, 32] ACT token IDs from the replay buffer."""
        features = self._preprocess_obs(x, action)
        logprobs, entropy, values = self.vla.evaluate_action(
            **features,
            unnorm_key=self.args.vla_unnorm_key
        )
        assert logprobs.shape[1] == 1 and entropy.shape[1] == 1 and values.shape[1] == 1
        return logprobs, entropy, values

    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.train()

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
        self.vla = UniVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )
        action_decoder_path = getattr(self.args, 'action_decoder_path', '')
        if not action_decoder_path:
            action_decoder_path = str(Path(self.args.vla_path) / "action_decoder.pt")
        self.vla.load_action_decoder(action_decoder_path)

        self.vla = PeftModel.from_pretrained(self.vla, path, is_trainable=True)
        if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
            ds = json.load(open(path / "dataset_statistics.json", "r"))
            self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        training_state = torch.load(path / "training_state.pt", map_location=self.tpdv["device"])
        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
        self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])


# Reuse OpenVLAPPO — the PPO algorithm is identical (operates on logprobs/entropy/values)
class UniVLALAMPPO:
    def __init__(self, all_args, policy: UniVLALAMPolicy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = 0.2
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv
        self.tpdv_vn = self.policy.tpdv_vn
        self.lambda_bc = self.args.lambda_bc

    def train_ppo_step(self, idx, total, batch, demo_batch=None):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)
        actions = torch.tensor(actions).to(self.tpdv["device"])  # [B, 32] ACT token IDs
        value_preds = torch.tensor(value_preds).to(**self.tpdv)
        returns = torch.tensor(returns).to(**self.tpdv_vn)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)
        returns_norm = returns.to(**self.tpdv)

        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)
        value_clip_ratio = ((value_pred_clipped - value_preds).abs() > self.ppo_clip).to(**self.tpdv).mean()
        value_loss = value_loss.mean()

        entropy_loss = entropy.mean()

        bc_loss = 0.0
        if demo_batch is not None:
            obs_demo_image, instruct_demo, actions_demo = demo_batch
            obs_demo = dict(image=torch.tensor(obs_demo_image).to(self.tpdv["device"]),
                            task_description=instruct_demo)
            actions_demo = torch.tensor(actions_demo).to(self.tpdv["device"])
            logprob_demo, _, _ = self.policy.evaluate_actions(obs_demo, actions_demo)
            bc_loss = -logprob_demo.mean()

        loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
        if demo_batch is not None:
            loss = loss + self.lambda_bc * bc_loss
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

        if demo_batch is not None:
            bc_loss = bc_loss.item()

        with open(self.args.log, "a") as f:
            f.write(f"[PPO Step {idx}/{total}] "
                    f"Returns: {returns.mean().item():.4f} | "
                    f"Policy Loss: {policy_loss.item():.4f} | "
                    f"Value Loss: {value_loss.item():.4f} | "
                    f"Entropy Loss: {entropy_loss.item():.4f} | "
                    f"BC Loss: {bc_loss:.4f} | "
                    f"Loss: {loss.item():.4f}\n")

        info = dict(
            loss=loss.item(), policy_loss=policy_loss.item(), value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(), bc_loss=bc_loss,
            ratio=ratio.mean().item(), ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),
            value_clip_ratio=value_clip_ratio.item(),
            value_old_mean=value_preds.mean().item(), values_mean=values.mean().item(),
            returns_mean=returns.mean().item(), returns_norm_mean=returns_norm.mean().item(),
            logprob_mean=logprob.mean().item(), logprob_old_mean=old_logprob.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()
        return info

    def train_ppo(self, buffer):
        train_info = defaultdict(lambda: [])
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        return {k: np.mean(v) for k, v in train_info.items()}
