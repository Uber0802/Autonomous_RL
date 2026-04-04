"""
UniVLA Policy Wrapper and PPO Algorithm for RL training.

Mirrors the OpenVLAPolicy / OpenVLAPPO interface so it can be dropped into
the existing AutoRL training loop with minimal changes.

Key differences from OpenVLA:
  - Vision encoding uses Emu3VisionVQ (discrete VQ tokens) instead of PrismaticImageProcessor
  - Text+vision tokens assembled via Emu3Processor.video_process(mode='VLA')
  - Action tokens mapped to the top of Emu3's 184k vocab (pad_token_id - 1 - bin_idx)
  - Uses ActionTokenizer (uniform 256 bins) for compatibility with simpler_wrapper
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor, AutoProcessor, BatchFeature

# Add Emu3 reference to path
_emu3_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'UniVLA', 'reference', 'Emu3')
if _emu3_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_emu3_path))

from emu3.mllm.processing_emu3 import Emu3Processor
from emu3.mllm import Emu3Tokenizer

# Add UniVLA models to path
_models_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'UniVLA', 'models')
if _models_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_models_path))

from modeling_emu3_rl import Emu3MoEForRL
from tokenizer.action_tokenizer import ActionTokenizer


def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class UniVLAPolicy:
    """
    UniVLA policy wrapper for RL training.

    Same interface as OpenVLAPolicy:
        - get_action(obs, deterministic) -> (values, action_token_ids, logprobs)
        - evaluate_actions(obs, action_token_ids) -> (logprobs, entropy, values)
        - get_value(obs) -> values
        - save(path) / load(path)
    """

    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.tpdv = dict(device=self.device, dtype=torch.bfloat16)
        self.tpdv_vn = dict(device=self.device, dtype=torch.float32)
        self.action_scale = 1.0

        # --- Paths ---
        vla_path = self.args.vla_path  # Path to UniVLA checkpoint
        # Vision VQ model path: either specified or same directory
        self.vision_vq_path = getattr(self.args, 'vision_vq_path', None) or vla_path

        # --- Load Emu3 tokenizer ---
        self.tokenizer = Emu3Tokenizer.from_pretrained(
            vla_path,
            padding_side="right",
            use_fast=False,
        )

        # --- Load VisionVQ for image encoding ---
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vision_vq_path, trust_remote_code=True
        )
        self.image_tokenizer = AutoModel.from_pretrained(
            self.vision_vq_path, trust_remote_code=True
        ).to(self.device).eval()
        # Freeze VisionVQ
        for param in self.image_tokenizer.parameters():
            param.requires_grad = False

        # --- Build Emu3Processor ---
        self.processor = Emu3Processor(
            self.image_processor,
            self.image_tokenizer,
            self.tokenizer,
        )

        # --- Action tokenizer (uniform 256 bins, same as OpenVLA's interface) ---
        self.n_action_bins = 256
        self.action_tokenizer = ActionTokenizer(
            self.tokenizer, bins=self.n_action_bins, min_action=-1, max_action=1
        )
        self.last_vocab_idx = self.action_tokenizer.last_vocab_idx  # pad_token_id - 1

        # eoa token id
        self.eoa_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eoa_token)

        # --- Load Emu3MoEForRL model ---
        self.vla = Emu3MoEForRL.from_pretrained(
            vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=f"cuda:{self.device_id}",
            vh_mode="a0",
        )

        # Setup action token range
        self.vla.setup_action_tokens(
            last_vocab_idx=self.last_vocab_idx,
            n_action_bins=self.n_action_bins,
            eoa_token_id=self.eoa_token_id,
        )

        # --- LoRA ---
        if not self.args.vla_load_path:
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=min(self.args.vla_lora_rank, 16),
                lora_dropout=0.0,
                target_modules=[
                    # Emu3 transformer layers
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "lm_head",
                ],
                init_lora_weights="gaussian",
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            self.vla = PeftModel.from_pretrained(
                self.vla, self.args.vla_load_path, is_trainable=True
            )
            print(f"UniVLA load: {self.args.vla_load_path}")

        # Set value head trainable
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        self.vla.print_trainable_parameters()

        # --- Load normalization stats ---
        self._load_norm_stats()

        # --- Optimizer ---
        self.params_vh = None
        self.params_vla = None
        self.vh_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        # --- Load training state if resuming ---
        if self.args.vla_load_path:
            training_state_path = Path(self.args.vla_load_path) / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=self.device)
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

    def _load_norm_stats(self):
        """Load action normalization statistics from UniVLA's normalizer configs."""
        unnorm_key = self.args.vla_unnorm_key
        # Try loading from UniVLA config directory
        norm_stats_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..', 'UniVLA', 'configs'
        )
        # Try bridge normalizer first
        for subdir in ['normalizer_bridge', 'normalizer_libero', 'normalizer_calvin']:
            path = os.path.join(norm_stats_dir, subdir, 'norm_stats.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                if "norm_stats" in data:
                    self.norm_stats = data["norm_stats"]
                    print(f"Loaded norm stats from {path}, keys: {list(self.norm_stats.keys())}")
                    return

        # Fallback: empty norm stats
        print("Warning: No norm_stats found, using empty dict")
        self.norm_stats = {}

    def get_action_stats(self):
        """Return normalization statistics for the environment wrapper."""
        unnorm_key = self.args.vla_unnorm_key
        if unnorm_key in self.norm_stats:
            return self.norm_stats[unnorm_key]
        # Return first available stats
        if self.norm_stats:
            key = list(self.norm_stats.keys())[0]
            print(f"Warning: unnorm_key '{unnorm_key}' not found, using '{key}'")
            return self.norm_stats[key]
        return None

    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        self.params_vla = [p for n, p in self.vla.named_parameters() if "value_head" not in n and p.requires_grad]
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB images to discrete VQ tokens using Emu3VisionVQ.

        Args:
            images: [B, H, W, C] uint8 tensor

        Returns:
            video_codes: list of [1, h, w] tensors (one per batch element)
        """
        from PIL import Image as PILImage

        batch_size = images.shape[0]
        video_codes_list = []

        # Process in batch: convert to PIL, then use image_processor + image_tokenizer
        pil_images = []
        for i in range(batch_size):
            img_np = images[i].cpu().numpy()  # [H, W, C]
            pil_images.append(PILImage.fromarray(img_np))

        # Batch encode through image processor
        pixel_values = self.image_processor(pil_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, self.image_tokenizer.dtype)

        # Encode to discrete tokens
        codes = self.image_tokenizer.encode(pixel_values)  # [B, h, w]

        # Return as list of [1, h, w] tensors for processor compatibility
        for i in range(batch_size):
            video_codes_list.append(codes[i].unsqueeze(0))  # [1, h, w]

        return video_codes_list

    def _preprocess_obs(self, x: dict, action: torch.Tensor = None) -> BatchFeature:
        """
        Preprocess observations into model inputs.

        Args:
            x: dict with "image" [B, H, W, C] uint8, "task_description" list[str]
            action: optional [B, action_dim] action token IDs for teacher-forced evaluation

        Returns:
            BatchFeature with input_ids, attention_mask (and labels if action provided)
        """
        images = x["image"]
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 4 and images.shape[3] == 3
        assert images.dtype == torch.uint8
        assert isinstance(task_description, list) and isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)

        batch_size = images.shape[0]

        # Step 1: Encode images to VQ tokens
        video_codes = self._encode_images(images)  # list of [1, h, w]

        # Step 2: Build prompts via Emu3Processor
        kwargs = dict(mode='VLA', padding="longest", return_tensors="pt")

        if action is None:
            # Inference: just prompt + vision + BOA token
            inputs = self.processor.video_process(
                text=task_description,
                video_tokens=video_codes,
                gripper_tokens=None,
                context_frames=1,
                frames=1,
                **kwargs,
            )
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            return BatchFeature(data=inputs)
        else:
            # Training: prompt + vision + BOA + action_tokens + EOA
            # First build base prompt (without action)
            inputs = self.processor.video_process(
                text=task_description,
                video_tokens=video_codes,
                gripper_tokens=None,
                context_frames=1,
                frames=1,
                **kwargs,
            )

            # Append action tokens + EOA to input_ids
            input_ids = inputs["input_ids"]  # [B, seq_len]
            attention_mask = inputs["attention_mask"]  # [B, seq_len]

            eoa_id = torch.tensor([[self.eoa_token_id]], device=input_ids.device).expand(batch_size, 1)
            # action: [B, action_dim] token IDs
            action_ids = action.to(input_ids.device).long()

            # Concatenate: [prompt + BOA] + [action_tokens] + [EOA]
            full_input_ids = torch.cat([input_ids, action_ids, eoa_id], dim=1)
            full_attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, action_ids.shape[1] + 1, dtype=attention_mask.dtype, device=attention_mask.device),
            ], dim=1)

            full_input_ids = full_input_ids.to(self.device)
            full_attention_mask = full_attention_mask.to(self.device)

            result = BatchFeature(data={
                "input_ids": full_input_ids,
                "attention_mask": full_attention_mask,
            })
            result["labels"] = full_input_ids.clone()

            return result

    def get_action(self, x: dict, deterministic) -> tuple:
        """
        Generate actions during rollout.

        Args:
            x: obs dict with "image" and "task_description"
            deterministic: if True use eval temperature

        Returns:
            (values [B,1], action_token_ids [B, 7], logprobs [B,1])
        """
        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        do_sample = (temperature != 0.0)
        features = self._preprocess_obs(x)

        torch.cuda.synchronize()
        values, action, logprobs = self.vla.predict_action_batch(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_action_tokens=7,  # 7 DoF for robot manipulation
            do_sample=do_sample,
            temperature=temperature,
        )

        assert len(values.shape) == 2 and values.shape[1] == 1
        assert len(action.shape) == 2 and action.shape[0] == values.shape[0]
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return values, action, logprobs

    def get_action_temp(self, x: dict, do_sample, temperature, num_beams=1) -> tuple:
        """Alternative action generation with explicit temperature."""
        features = self._preprocess_obs(x)

        _, action, logprobs = self.vla.predict_action_batch(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_action_tokens=7,
            do_sample=do_sample,
            temperature=temperature,
        )

        assert len(action.shape) == 2
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return action, logprobs

    def get_value(self, x: dict) -> torch.Tensor:
        """Value-only forward pass."""
        features = self._preprocess_obs(x)

        value = self.vla.get_value(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
        )

        assert len(value.shape) == 2 and value.shape[1] == 1
        return value

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple:
        """
        Teacher-forced evaluation of given actions.

        Args:
            x: obs dict
            action: [B, 7] action token IDs

        Returns:
            (logprobs [B,1], entropy [B,1], values [B,1])
        """
        features = self._preprocess_obs(x, action)

        logprobs, entropy, values = self.vla.evaluate_action(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            action_token_ids=action.to(self.device).long(),
            action_len=action.shape[1],
        )

        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1
        assert len(entropy.shape) == 2 and entropy.shape[1] == 1
        assert len(values.shape) == 2 and values.shape[1] == 1

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

        # Save norm stats
        json.dump(self.norm_stats, open(path / "dataset_statistics.json", "w"))

    def load(self, path: Path):
        del self.vla
        torch.cuda.empty_cache()

        self.vla = Emu3MoEForRL.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=f"cuda:{self.device_id}",
            vh_mode="a0",
        )
        self.vla.setup_action_tokens(
            last_vocab_idx=self.last_vocab_idx,
            n_action_bins=self.n_action_bins,
            eoa_token_id=self.eoa_token_id,
        )
        self.vla = PeftModel.from_pretrained(self.vla, path, is_trainable=True)
        self.vla.print_trainable_parameters()

        # Reload norm stats
        stats_path = path / "dataset_statistics.json"
        if stats_path.exists():
            self.norm_stats = json.load(open(stats_path, "r"))

        training_state_path = path / "training_state.pt"
        training_state = torch.load(training_state_path, map_location=self.device)

        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
        else:
            print("Warning: value_head state not found in training_state")

        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
        self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])


class UniVLAPPO:
    """
    PPO algorithm for UniVLA.
    Identical logic to OpenVLAPPO — only the policy type changes.
    """

    def __init__(self, all_args, policy: UniVLAPolicy):
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
        actions = torch.tensor(actions).to(self.tpdv["device"])
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
            actions_demo = torch.tensor(actions_demo).to(self.tpdv["device"])
            logprob_demo, _, _ = self.policy.evaluate_actions(obs_demo, actions_demo)
            bc_loss = -logprob_demo.mean()

        # Total loss
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
        log_str = (
            f"[PPO Step {idx}/{total}] "
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
        actions = torch.tensor(actions).to(self.tpdv["device"])
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
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)
        return final_info

    def train_ppo_joint(self, buffer, buffer2):
        train_info = defaultdict(lambda: [])
        buffer.cat_buffer(buffer2)
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train joint buffer"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)
        return final_info

    def train_ppo_2buffer(self, buffer1, buffer2):
        train_info = defaultdict(lambda: [])
        buffer1.compute_returns_ppo()
        buffer2.compute_returns_ppo()
        all_adv = np.concatenate([buffer1.advantages, buffer2.advantages])
        mean_adv = all_adv.mean()
        std_adv = all_adv.std()
        buffer1.advantages = (buffer1.advantages - mean_adv) / (std_adv + 1e-5)
        buffer2.advantages = (buffer2.advantages - mean_adv) / (std_adv + 1e-5)
        assert buffer1.get_minibatch_count() == buffer2.get_minibatch_count()
        minibatch_count = buffer1.get_minibatch_count() + buffer2.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator1 = buffer1.feed_forward_generator()
            data_generator2 = buffer2.feed_forward_generator()
            for idx in tqdm(range(minibatch_count), desc="train interleaved"):
                if idx % 2 == 0:
                    batch = next(data_generator1)
                else:
                    batch = next(data_generator2)
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)
        return final_info

    def train_grpo(self, buffer):
        train_info = defaultdict(lambda: [])
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
