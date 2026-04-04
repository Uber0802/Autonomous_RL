"""
cogact_train.py

CogACTPolicy and CogACTPPO — CogACT VLM with Gaussian action head for PPO training.
Mirrors the interface of OpenVLAPolicy / OpenVLAPPO in openvla_train.py.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import LlamaTokenizerFast

from vla.load import load_vla
from simpler_env.policies.cogact.cogact_model import GaussianActionHead, ValueHead


def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class CogACTPolicy:
    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self._half_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tpdv = dict(device=self.device, dtype=self._half_dtype)
        self.tpdv_vn = dict(device=self.device, dtype=torch.float32)

        # Load CogACT model (VLM + DiT action head, but we only use VLM)
        self.cogact = load_vla(
            self.args.vla_path,
            load_for_training=True,
            action_model_type=getattr(self.args, 'cogact_action_model_type', 'DiT-B'),
            future_action_window_size=getattr(self.args, 'cogact_future_window', 15),
            action_dim=7,
        )
        self.cogact = self.cogact.to(self.device)

        # Extract VLM components for later use
        self.vlm = self.cogact.vlm
        self.image_transform = self.vlm.vision_backbone.image_transform
        self.tokenizer = self.vlm.llm_backbone.tokenizer
        self.norm_stats = self.cogact.norm_stats
        self.hidden_dim = self.vlm.llm_backbone.llm.lm_head.in_features  # 4096

        # Freeze VLM base params (LoRA will be added separately)
        for p in self.vlm.parameters():
            p.requires_grad = False
        # Also freeze the DiT action model — we won't use it
        for p in self.cogact.action_model.parameters():
            p.requires_grad = False

        # Add Gaussian action head and value head
        self.action_head = GaussianActionHead(
            hidden_dim=self.hidden_dim,
            action_dim=7,
        ).to(**self.tpdv)
        self.value_head = ValueHead(hidden_dim=self.hidden_dim).to(**self.tpdv)

        # Load BC-pretrained weights if available (distilled from DiT)
        bc_init_path = getattr(self.args, 'bc_init_path', None)
        if bc_init_path and os.path.exists(bc_init_path):
            bc_state = torch.load(bc_init_path, map_location=self.device)
            self.action_head.load_state_dict(bc_state["action_head"])
            if "value_head" in bc_state:
                self.value_head.load_state_dict(bc_state["value_head"])
            print(f"BC-pretrained heads loaded from {bc_init_path} "
                  f"(MSE={bc_state.get('final_mse', '?'):.6f})")

        # Setup LoRA on VLM (same targets as AutoRL's OpenVLA)
        if not getattr(self.args, 'vla_load_path', None):
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=min(self.args.vla_lora_rank, 16),
                lora_dropout=0.0,
                target_modules=[
                    "proj", "qkv", "fc1", "fc2",       # vision
                    "q", "kv", "fc3",                    # projection
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", "lm_head",  # LLM
                ],
                init_lora_weights="gaussian",
            )
            self.vlm = get_peft_model(self.vlm, lora_config)
        else:
            self.vlm = PeftModel.from_pretrained(self.vlm, self.args.vla_load_path, is_trainable=True)
            print(f"CogACT VLM LoRA load: {self.args.vla_load_path}")

        self.vlm.print_trainable_parameters()

        # Optimizers (same as AutoRL: separate lr for value head vs LoRA+action head)
        self.params_vh = list(self.value_head.parameters())
        self.params_vla = (
            list(self.action_head.parameters()) +
            [p for p in self.vlm.parameters() if p.requires_grad]
        )
        self.vh_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        # Load training state if resuming
        if getattr(self.args, 'vla_load_path', None):
            self._load_training_state(Path(self.args.vla_load_path))

    def _setup_optimizer(self):
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)

    def _load_training_state(self, path: Path):
        training_state_path = path / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(training_state_path, map_location=self.device)
            if "action_head" in state:
                self.action_head.load_state_dict(state["action_head"])
            if "value_head" in state:
                self.value_head.load_state_dict(state["value_head"])
            self._setup_optimizer()
            if "vh_optimizer" in state:
                self.vh_optimizer.load_state_dict(state["vh_optimizer"])
            if "vla_optimizer" in state:
                self.vla_optimizer.load_state_dict(state["vla_optimizer"])
            print(f"Training state loaded from {training_state_path}")
        else:
            print(f"Warning: no training_state.pt found at {path}")

    def _get_num_patches(self):
        """Get number of vision patches to skip when extracting cognition token."""
        vb = self.vlm.base_model if hasattr(self.vlm, 'base_model') else self.vlm
        # Navigate through PeftModel wrapper if present
        while hasattr(vb, 'model'):
            if hasattr(vb, 'vision_backbone'):
                break
            vb = vb.model
        if hasattr(vb, 'vision_backbone'):
            vis = vb.vision_backbone
        else:
            vis = self.cogact.vlm.vision_backbone
        if vis.featurizer is not None:
            return vis.featurizer.patch_embed.num_patches
        elif hasattr(vis, 'siglip_featurizer') and vis.siglip_featurizer is not None:
            return vis.siglip_featurizer.patch_embed.num_patches
        raise ValueError("No vision backbone found")

    def _extract_cognition(self, input_ids, attention_mask, pixel_values):
        """
        Run VLM forward pass and extract cognition token [B, hidden_dim].
        Uses the same logic as CogACT's cogactvla.py forward().
        """
        autocast_dtype = self._half_dtype
        with torch.autocast("cuda", dtype=autocast_dtype,
                            enabled=self.cogact.vlm.enable_mixed_precision_training):
            output = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract cognition feature (last valid token of last hidden layer)
        num_patch = self._get_num_patches()
        last_hidden = output.hidden_states[-1]  # [B, seq_len, 4096]
        last_hidden = last_hidden[:, num_patch:]

        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
        cognition = last_hidden.gather(1, expanded_indices.unsqueeze(1)).squeeze(1)  # [B, 4096]

        return cognition

    def _preprocess_obs(self, x: dict):
        """
        Convert env observations to CogACT VLM inputs.
        Input: x = {"image": [B, H, W, 3] uint8 tensor, "task_description": list[str]}
        Returns: input_ids, attention_mask, pixel_values
        """
        images = x["image"]           # [B, H, W, 3] uint8
        task_descriptions = x["task_description"]  # list[str]

        B = images.shape[0]
        assert images.dtype == torch.uint8

        # Build prompts and tokenize (same format as CogACT predict_action_batch)
        input_ids_list = []
        pixel_values_list = []

        for i in range(B):
            prompt_builder = self.cogact.vlm.get_prompt_builder()
            prompt_builder.add_turn(
                role="human",
                message=f"What action should the robot take to {task_descriptions[i].lower()}?",
            )
            prompt_text = prompt_builder.get_prompt()

            single_ids = self.tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.squeeze(0)
            # Append special tokens: empty (29871) + EOS (2), matching CogACT training format
            single_ids = torch.cat([single_ids, torch.tensor([29871, 2], dtype=torch.long)])
            input_ids_list.append(single_ids.to(self.device))

            # Process image: convert uint8 [H,W,3] numpy/tensor to PIL then transform
            from PIL import Image
            img_np = images[i].cpu().numpy() if isinstance(images[i], torch.Tensor) else images[i]
            pil_img = Image.fromarray(img_np)
            pixel_values_list.append(self.image_transform(pil_img))

        # Pad input_ids
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(pad_token_id).to(self.device)
        input_ids = input_ids.to(self.device)

        # Stack pixel values
        if isinstance(pixel_values_list[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values_list).to(self.device)
        elif isinstance(pixel_values_list[0], dict):
            pixel_values = {
                k: torch.stack([pv[k] for pv in pixel_values_list]).to(self.device)
                for k in pixel_values_list[0]
            }
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values_list[0])}")

        return input_ids, attention_mask, pixel_values

    @torch.no_grad()
    def get_action(self, x: dict, deterministic: bool) -> tuple:
        """
        Args:
            x: {"image": [B,H,W,3] uint8, "task_description": list[str]}
            deterministic: bool
        Returns:
            values: [B, 1]
            actions: [B, 7] (continuous, normalized [-1, 1])
            logprobs: [B, 1]
        """
        input_ids, attention_mask, pixel_values = self._preprocess_obs(x)
        cognition = self._extract_cognition(input_ids, attention_mask, pixel_values)
        cognition = cognition.to(**self.tpdv)

        actions, logprobs = self.action_head.get_action(cognition, deterministic=deterministic)
        values = self.value_head(cognition)

        # Clamp actions to [-1, 1] for safety
        actions = actions.clamp(-1, 1)

        assert values.shape == (actions.shape[0], 1)
        assert logprobs.shape == (actions.shape[0], 1)

        return values, actions, logprobs

    def evaluate_actions(self, x: dict, actions: torch.Tensor) -> tuple:
        """
        Recompute log_prob and entropy for given actions (used in PPO update).
        Args:
            x: {"image": [B,H,W,3] uint8, "task_description": list[str]}
            actions: [B, 7] continuous actions
        Returns:
            logprobs: [B, 1]
            entropy: [B, 1]
            values: [B, 1]
        """
        input_ids, attention_mask, pixel_values = self._preprocess_obs(x)
        cognition = self._extract_cognition(input_ids, attention_mask, pixel_values)
        cognition = cognition.to(**self.tpdv)

        logprobs, entropy = self.action_head.evaluate(cognition, actions.to(**self.tpdv))
        values = self.value_head(cognition)

        return logprobs, entropy, values

    @torch.no_grad()
    def get_value(self, x: dict) -> torch.Tensor:
        """Returns values [B, 1]."""
        input_ids, attention_mask, pixel_values = self._preprocess_obs(x)
        cognition = self._extract_cognition(input_ids, attention_mask, pixel_values)
        cognition = cognition.to(**self.tpdv)
        return self.value_head(cognition)

    def prep_rollout(self):
        self.vlm.eval()
        self.action_head.eval()
        self.value_head.eval()

    def prep_training(self):
        self.vlm.train()
        self.action_head.train()
        self.value_head.train()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.vlm.save_pretrained(str(path))

        # Save action head, value head, and optimizer states
        training_state = {
            "action_head": self.action_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        torch.save(training_state, path / "training_state.pt")

        # Save norm stats
        if self.norm_stats:
            json.dump(self.norm_stats, open(path / "dataset_statistics.json", "w"))

    def load(self, path: Path):
        # Reload LoRA weights
        del self.vlm
        torch.cuda.empty_cache()

        # Re-load base VLM
        self.cogact = load_vla(
            self.args.vla_path,
            load_for_training=True,
            action_model_type=getattr(self.args, 'cogact_action_model_type', 'DiT-B'),
            future_action_window_size=getattr(self.args, 'cogact_future_window', 15),
            action_dim=7,
        )
        self.cogact = self.cogact.to(self.device)
        self.vlm = self.cogact.vlm

        for p in self.vlm.parameters():
            p.requires_grad = False
        for p in self.cogact.action_model.parameters():
            p.requires_grad = False

        # Load LoRA
        self.vlm = PeftModel.from_pretrained(self.vlm, path, is_trainable=True)
        self.vlm.print_trainable_parameters()

        # Load training state
        self._load_training_state(path)


class CogACTPPO:
    """PPO trainer for CogACTPolicy. Same interface as OpenVLAPPO."""

    def __init__(self, all_args, policy: CogACTPolicy):
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
        actions = torch.tensor(actions).to(**self.tpdv)  # float32 continuous actions
        value_preds = torch.tensor(value_preds).to(**self.tpdv)
        returns = torch.tensor(returns).to(**self.tpdv_vn)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)
        returns_norm = returns.to(**self.tpdv)

        # Policy loss
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

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

        # Entropy loss
        entropy_loss = entropy.mean()

        # Behavior Cloning Loss
        bc_loss = 0.0
        if demo_batch is not None:
            obs_demo_image, instruct_demo, actions_demo = demo_batch
            obs_demo = dict(image=torch.tensor(obs_demo_image).to(self.tpdv["device"]),
                            task_description=instruct_demo)
            actions_demo = torch.tensor(actions_demo).to(**self.tpdv)
            logprob_demo, _, _ = self.policy.evaluate_actions(obs_demo, actions_demo)
            bc_loss = -logprob_demo.mean()

        # Total loss
        loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
        if demo_batch is not None:
            loss = loss + self.lambda_bc * bc_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.params_vla + self.policy.params_vh, self.ppo_grad_norm
            )
            self.policy.vh_optimizer.step()
            self.policy.vla_optimizer.step()
            self.policy.vh_optimizer.zero_grad()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        if demo_batch is not None:
            bc_loss = bc_loss.item()

        log_str = (
            f"[CogACT PPO Step {idx}/{total}] "
            f"Returns: {returns.mean().item():.4f} | "
            f"Advantages: {advantages.mean().item():.4f} | "
            f"Policy Loss: {policy_loss.item():.4f} | "
            f"Value Loss: {value_loss.item():.4f} | "
            f"Entropy Loss: {entropy_loss.item():.4f} | "
            f"BC Loss: {bc_loss:.4f} | "
            f"Loss: {loss.item():.4f}\n"
        )
        with open(self.args.log, "a") as f:
            f.write(log_str)

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            bc_loss=bc_loss,
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

    def train_grpo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)
        actions = torch.tensor(actions).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)

        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        entropy_loss = entropy.mean()

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

    def train_ppo(self, buffer):
        train_info = defaultdict(lambda: [])
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="CogACT PPO train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        return {k: np.mean(v) for k, v in train_info.items()}

    def train_ppo_joint(self, buffer, buffer2):
        train_info = defaultdict(lambda: [])
        buffer.cat_buffer(buffer2)
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="CogACT PPO train joint"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        return {k: np.mean(v) for k, v in train_info.items()}

    def train_ppo_2buffer(self, buffer1, buffer2):
        train_info = defaultdict(lambda: [])
        buffer1.compute_returns_ppo()
        buffer2.compute_returns_ppo()

        all_adv = np.concatenate([buffer1.advantages, buffer2.advantages])
        mean_adv, std_adv = all_adv.mean(), all_adv.std()
        buffer1.advantages = (buffer1.advantages - mean_adv) / (std_adv + 1e-5)
        buffer2.advantages = (buffer2.advantages - mean_adv) / (std_adv + 1e-5)
        assert buffer1.get_minibatch_count() == buffer2.get_minibatch_count()
        minibatch_count = buffer1.get_minibatch_count() + buffer2.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            gen1 = buffer1.feed_forward_generator()
            gen2 = buffer2.feed_forward_generator()
            for idx in tqdm(range(minibatch_count), desc="CogACT PPO train interleaved"):
                batch = next(gen1) if idx % 2 == 0 else next(gen2)
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        return {k: np.mean(v) for k, v in train_info.items()}

    def train_grpo(self, buffer):
        train_info = defaultdict(lambda: [])
        buffer.compute_returns_grpo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="CogACT GRPO train"):
                info = self.train_grpo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        return {k: np.mean(v) for k, v in train_info.items()}
