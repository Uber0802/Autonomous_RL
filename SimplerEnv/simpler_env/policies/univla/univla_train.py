"""
UniVLA (Emu3) Policy Wrapper and PPO Algorithm for RL training.

Uses `Yuqi1997/UniVLA/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K` as the VLA model with:
  - Emu3VisionVQ for vision tokenization (discrete VQ codes)
  - Emu3Processor.video_process(mode='VLA') for prompt assembly
  - FAST BPE action tokenizer (fast-bridge-t5-s50) — variable-length output
  - Action decoding: FAST BPE → DCT coeffs → IDCT → [10, 7] trajectory → first step
  - Log-prob aggregation over variable-length token sequences (including EOA)

Interface matches OpenVLAPolicy / UniVLALAMPolicy so it drops into the existing
AutoRL training loop via `--vla_type univla_emu3`. The policy returns continuous
actions directly (via `SimlerWrapper.step_continuous`), similar to the LAM path.
"""

import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModel, AutoImageProcessor, AutoProcessor, BatchFeature

# Add Emu3 reference and UniVLA models to path
_here = os.path.dirname(__file__)
_emu3_path = os.path.abspath(os.path.join(_here, '..', '..', '..', '..', 'UniVLA', 'reference', 'Emu3'))
if _emu3_path not in sys.path:
    sys.path.insert(0, _emu3_path)
_models_path = os.path.abspath(os.path.join(_here, '..', '..', '..', '..', 'UniVLA', 'models'))
if _models_path not in sys.path:
    sys.path.insert(0, _models_path)

from emu3.mllm.processing_emu3 import Emu3Processor
from emu3.mllm import Emu3Tokenizer

from modeling_emu3_rl import Emu3MoEForRL


def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class UniVLAPolicy:
    """Emu3-based UniVLA policy wrapper for PPO training.

    Interface contract:
        get_action(obs, deterministic) ->
            (values [B,1], continuous_actions [B,7], padded_tokens [B,max_A],
             lengths [B], logprobs [B,1])
        evaluate_actions(obs, padded_tokens, lengths) ->
            (logprobs [B,1], entropy [B,1], values [B,1])
        get_value(obs) -> values [B,1]
        save(path) / load(path)

    The training loop treats this as the LAM path (continuous actions fed to
    env via `step_continuous`).
    """

    # Action-tokenizer constants (fixed by the pretrained checkpoint)
    MAX_ACTION_TOKENS = 50
    ACTION_CHUNK_SIZE = 10
    ACTION_DIM = 7

    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.tpdv = dict(device=self.device, dtype=torch.bfloat16)
        self.tpdv_vn = dict(device=self.device, dtype=torch.float32)
        self.action_scale = 1.0

        vla_path = self.args.vla_path
        vision_vq_path = getattr(self.args, 'vision_vq_path', '') or \
            os.path.join(os.path.dirname(vla_path), '..', 'emu3-vision-tokenizer')
        vision_vq_path = os.path.abspath(vision_vq_path)
        fast_path = getattr(self.args, 'fast_tokenizer_path', '') or \
            os.path.join(os.path.dirname(vla_path), '..', 'fast-bridge-t5-s50')
        fast_path = os.path.abspath(fast_path)

        print(f"[UniVLAPolicy] vla_path={vla_path}")
        print(f"[UniVLAPolicy] vision_vq_path={vision_vq_path}")
        print(f"[UniVLAPolicy] fast_path={fast_path}")

        # --- Emu3 text tokenizer (for building prompts) ---
        self.tokenizer = Emu3Tokenizer.from_pretrained(
            vla_path, padding_side="right", use_fast=False,
        )

        # --- Emu3 VisionVQ encoder (frozen) ---
        # The pretraining used ~4800 tokens per image (60×80 grid, 512×512 input).
        # For RL training we need a smaller grid to fit backprop in memory.
        # Target: ~972 tokens (27×36 grid, 216×288 input area).
        self.image_processor = AutoImageProcessor.from_pretrained(
            vision_vq_path, trust_remote_code=True,
        )
        target_pixels = getattr(self.args, 'vla_image_pixels', 256 * 256)
        self.image_processor.min_pixels = target_pixels
        self.image_processor.max_pixels = target_pixels
        self.image_processor.size = {"min_pixels": target_pixels, "max_pixels": target_pixels}
        self.image_tokenizer = AutoModel.from_pretrained(
            vision_vq_path, trust_remote_code=True,
        ).to(self.device).eval()
        for p in self.image_tokenizer.parameters():
            p.requires_grad = False

        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer,
        )

        # --- FAST action tokenizer (BPE over DCT coefficients) ---
        self.fast_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)
        # Freeze to time_horizon=10, action_dim=7 (Bridge-specific; matches training)
        self.fast_tokenizer.called_time_horizon = self.ACTION_CHUNK_SIZE
        self.fast_tokenizer.called_action_dim = self.ACTION_DIM

        # --- Action vocab mapping ---
        self.last_vocab_idx = self.tokenizer.pad_token_id - 1  # 151642
        self.fast_vocab_size = self.fast_tokenizer.vocab_size  # 1024
        self.eoa_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eoa_token)  # 151845

        # --- Load the RL-adapted Emu3MoE model ---
        self.vla = Emu3MoEForRL.from_pretrained(
            vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            vh_mode="a0",
        ).to(self.device)

        self.vla.setup_action_tokens(
            last_vocab_idx=self.last_vocab_idx,
            fast_vocab_size=self.fast_vocab_size,
            eoa_token_id=self.eoa_token_id,
        )

        # Enable gradient checkpointing to halve backprop memory cost
        # (at ~30% compute cost). Essential for Emu3 because the prompt has
        # ~1000 tokens even with reduced vision resolution.
        self.vla.gradient_checkpointing_enable()
        self.vla.enable_input_require_grads()

        # --- LoRA ---
        if not getattr(self.args, 'vla_load_path', None):
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=min(self.args.vla_lora_rank, 16),
                lora_dropout=0.0,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                init_lora_weights="gaussian",
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            self.vla = PeftModel.from_pretrained(
                self.vla, self.args.vla_load_path, is_trainable=True,
            )
            print(f"UniVLA load: {self.args.vla_load_path}")

        # Value head is always trainable
        for name, p in self.vla.named_parameters():
            if "value_head" in name:
                p.requires_grad = True

        self.vla.print_trainable_parameters()

        # --- Load normalization stats ---
        self._load_norm_stats()

        # --- Optimizer ---
        self.params_vh = None
        self.params_vla = None
        self.vh_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        # --- Resume training state ---
        if getattr(self.args, 'vla_load_path', None):
            ts_path = Path(self.args.vla_load_path) / "training_state.pt"
            if ts_path.exists():
                ts = torch.load(ts_path, map_location=self.device)
                if "vh" in ts:
                    self.vla.value_head.load_state_dict(ts['vh'], assign=True)
                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(ts['vh_optimizer'])
                self.vla_optimizer.load_state_dict(ts['vla_optimizer'])

    # ------------------------------------------------------------------
    # Normalization stats
    # ------------------------------------------------------------------
    def _load_norm_stats(self):
        """Load bridge_robot norm stats directly from the Emu3 checkpoint."""
        ns_path = os.path.join(self.args.vla_path, 'norm_stats.json')
        if os.path.exists(ns_path):
            with open(ns_path) as f:
                self.norm_stats = json.load(f)["norm_stats"]
            print(f"[UniVLAPolicy] Loaded norm_stats from {ns_path}, keys: {list(self.norm_stats.keys())}")
        else:
            print(f"[UniVLAPolicy] WARNING: no norm_stats.json at {ns_path}")
            self.norm_stats = {}

        # Pre-compute q01/q99 tensors on device.
        # norm_stats.json layout: {"bridge_robot": {"q01": [...], "q99": [...], ...}}
        key = getattr(self.args, 'vla_unnorm_key', None) or next(iter(self.norm_stats.keys()))
        if key not in self.norm_stats:
            print(f"[UniVLAPolicy] unnorm_key '{key}' not in {list(self.norm_stats.keys())}; using first.")
            key = next(iter(self.norm_stats.keys()))
        stats = self.norm_stats[key]
        self.q01 = torch.tensor(stats['q01'], device=self.device, dtype=torch.float32)
        self.q99 = torch.tensor(stats['q99'], device=self.device, dtype=torch.float32)
        self._unnorm_key = key

    def get_action_stats(self):
        """Expose stats for the env wrapper (not used in this path — env gets continuous actions)."""
        return self.norm_stats.get(self._unnorm_key, None)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        self.params_vla = [p for n, p in self.vla.named_parameters() if "value_head" not in n and p.requires_grad]
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)

    # ------------------------------------------------------------------
    # Image encoding (no grad)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """RGB [B, H, W, 3] uint8 → video codes [B, 1, h, w]."""
        from PIL import Image as PILImage

        B = images.shape[0]
        pil_images = [PILImage.fromarray(images[i].cpu().numpy()) for i in range(B)]
        pixel_values = self.image_processor(pil_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, self.image_tokenizer.dtype)
        codes = self.image_tokenizer.encode(pixel_values)  # [B, h, w]
        # Add frames dim: [B, 1, h, w]
        if codes.ndim == 3:
            codes = codes.unsqueeze(1)
        return codes

    # ------------------------------------------------------------------
    # Prompt building (prompt-only, no teacher-forced action)
    # ------------------------------------------------------------------
    def _build_prompts(self, images: torch.Tensor, task_descriptions) -> dict:
        """Build model inputs from a batch of observations (inference mode)."""
        assert isinstance(images, torch.Tensor)
        assert images.ndim == 4 and images.shape[3] == 3 and images.dtype == torch.uint8
        B = images.shape[0]
        assert isinstance(task_descriptions, list) and len(task_descriptions) == B

        video_codes = self._encode_images(images)  # [B, 1, h, w]

        features = self.processor.video_process(
            text=task_descriptions,
            video_tokens=video_codes,
            gripper_tokens=None,
            context_frames=1,
            frames=1,
            return_tensors="pt",
            mode="VLA",
            padding="longest",
        )
        return {
            "input_ids": features["input_ids"].to(self.device),
            "attention_mask": features["attention_mask"].to(self.device),
        }

    # ------------------------------------------------------------------
    # FAST decode: BPE token ids → continuous [B, 7] action
    # ------------------------------------------------------------------
    def _decode_actions(
        self,
        padded_tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """FAST BPE → IDCT → unnormalize → first-step continuous action [B, 7]."""
        B = padded_tokens.shape[0]
        token_lists = []
        for b in range(B):
            L = int(lengths[b].item())
            bpe_ids = (self.last_vocab_idx - padded_tokens[b, :L]).cpu().tolist()
            token_lists.append(bpe_ids)

        # Decode: list[list[int]] → np.array [B, 10, 7] normalized
        try:
            decoded = self.fast_tokenizer.decode(
                token_lists,
                time_horizon=self.ACTION_CHUNK_SIZE,
                action_dim=self.ACTION_DIM,
            )
        except Exception as e:
            print(f"[UniVLAPolicy] FAST decode error: {e} — falling back to zeros")
            decoded = np.zeros((B, self.ACTION_CHUNK_SIZE, self.ACTION_DIM), dtype=np.float32)

        decoded = np.asarray(decoded, dtype=np.float32)  # [B, 10, 7]
        first_step = decoded[:, 0, :]  # [B, 7]
        first_step_t = torch.tensor(first_step, device=self.device, dtype=torch.float32)

        # Unnormalize from [-1, 1] to physical units using q01/q99
        phys = 0.5 * (first_step_t + 1.0) * (self.q99 - self.q01) + self.q01  # [B, 7]
        return phys

    # ------------------------------------------------------------------
    # Rollout-time action prediction
    # ------------------------------------------------------------------
    def _compute_lengths(self, padded_tokens: torch.Tensor) -> torch.Tensor:
        """Recover per-sample lengths by finding the first EOA in padded_tokens."""
        B = padded_tokens.shape[0]
        lengths = torch.full((B,), padded_tokens.shape[1], dtype=torch.long, device=padded_tokens.device)
        for b in range(B):
            eoa_pos = (padded_tokens[b] == self.eoa_token_id).nonzero(as_tuple=True)[0]
            if len(eoa_pos) > 0:
                lengths[b] = eoa_pos[0].item()
        return lengths

    def get_action(self, x: dict, deterministic) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate an action. Interface matches the LAM path (4-tuple).

        Returns:
            values             [B, 1]
            continuous_actions [B, 7]   — unnormalized, direct env input
            padded_tokens      [B, max_A]  — includes trailing EOA pads
            logprobs           [B, 1]
        """
        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        do_sample = (temperature != 0.0)
        features = self._build_prompts(x["image"], x["task_description"])

        values, padded_tokens, lengths, logprobs = self.vla.predict_action_batch(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_action_tokens=self.MAX_ACTION_TOKENS,
            do_sample=do_sample,
            temperature=temperature,
        )
        continuous_actions = self._decode_actions(padded_tokens, lengths)

        assert values.shape == (features["input_ids"].shape[0], 1)
        assert padded_tokens.shape[1] == self.MAX_ACTION_TOKENS
        assert logprobs.shape == (features["input_ids"].shape[0], 1)
        return values, continuous_actions, padded_tokens, logprobs

    def get_action_temp(self, x: dict, do_sample, temperature, num_beams=1):
        features = self._build_prompts(x["image"], x["task_description"])
        _, padded_tokens, lengths, logprobs = self.vla.predict_action_batch(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_action_tokens=self.MAX_ACTION_TOKENS,
            do_sample=do_sample,
            temperature=temperature,
        )
        continuous_actions = self._decode_actions(padded_tokens, lengths)
        return continuous_actions, padded_tokens, logprobs

    # ------------------------------------------------------------------
    # Value-only forward pass
    # ------------------------------------------------------------------
    def get_value(self, x: dict) -> torch.Tensor:
        features = self._build_prompts(x["image"], x["task_description"])
        value = self.vla.get_value(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
        )
        assert value.shape[1] == 1
        return value

    # ------------------------------------------------------------------
    # Teacher-forced evaluation (for PPO policy update)
    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        x: dict,
        padded_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced evaluation of stored padded action tokens.

        Lengths are recomputed from EOA positions (no explicit length buffer).

        Args:
            x: obs dict with "image", "task_description"
            padded_tokens: [B, max_A] Long
        Returns:
            logprobs [B,1], entropy [B,1], values [B,1]
        """
        features = self._build_prompts(x["image"], x["task_description"])
        prompt_ids = features["input_ids"]
        prompt_attn = features["attention_mask"]
        B, prompt_len = prompt_ids.shape

        padded_tokens = padded_tokens.to(self.device).long()
        lengths = self._compute_lengths(padded_tokens)

        # Build full sequence = prompt + padded_tokens
        full_ids = torch.cat([prompt_ids, padded_tokens], dim=1)
        # Build attention mask: prompt_attn + 1s up to (length + 1) (include EOA), 0 beyond
        tok_attn = torch.zeros_like(padded_tokens)
        for b in range(B):
            valid = min(int(lengths[b].item()) + 1, padded_tokens.shape[1])
            tok_attn[b, :valid] = 1
        full_attn = torch.cat([prompt_attn, tok_attn], dim=1)

        logprobs, entropy, values = self.vla.evaluate_action(
            input_ids=full_ids,
            attention_mask=full_attn,
            action_tokens=padded_tokens,
            action_lengths=lengths,
        )
        assert logprobs.shape[1] == 1 and entropy.shape[1] == 1 and values.shape[1] == 1
        return logprobs, entropy, values

    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.train()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.vla.save_pretrained(str(path))
        training_state = {
            "vh": self.vla.value_head.state_dict() if hasattr(self.vla, 'value_head') else self.vla.base_model.model.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        torch.save(training_state, path / "training_state.pt")
        with open(path / "dataset_statistics.json", "w") as f:
            json.dump(self.norm_stats, f)

    def load(self, path: Path):
        del self.vla
        torch.cuda.empty_cache()
        self.vla = Emu3MoEForRL.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            vh_mode="a0",
        ).to(self.device)
        self.vla.setup_action_tokens(
            last_vocab_idx=self.last_vocab_idx,
            fast_vocab_size=self.fast_vocab_size,
            eoa_token_id=self.eoa_token_id,
        )
        self.vla = PeftModel.from_pretrained(self.vla, path, is_trainable=True)
        self.vla.print_trainable_parameters()

        stats_path = path / "dataset_statistics.json"
        if stats_path.exists():
            self.norm_stats = json.load(open(stats_path, "r"))
        ts_path = path / "training_state.pt"
        ts = torch.load(ts_path, map_location=self.device)
        if "vh" in ts:
            try:
                self.vla.value_head.load_state_dict(ts['vh'], assign=True)
            except AttributeError:
                self.vla.base_model.model.value_head.load_state_dict(ts['vh'], assign=True)
        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(ts['vh_optimizer'])
        self.vla_optimizer.load_state_dict(ts['vla_optimizer'])


class UniVLAPPO:
    """PPO algorithm — identical logic to OpenVLAPPO, adapted for padded+length inputs."""

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

    def train_ppo(self, buffer):
        """Orchestrator for PPO update (matches OpenVLAPPO.train_ppo)."""
        from collections import defaultdict
        from tqdm import tqdm as _tqdm
        train_info = defaultdict(lambda: [])

        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()
            for idx, batch in _tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)
        return final_info

    def train_ppo_step(self, idx, total, batch, demo_batch=None):
        """Process a PPO mini-batch. Matches the LAM path's 8-tuple layout.

        batch = (obs_image, instruct, padded_tokens, value_preds, returns, masks, old_logprob, advantages)
        """
        obs_image, instruct, padded_tokens, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)
        padded_tokens = torch.tensor(padded_tokens).to(self.tpdv["device"]).long()
        value_preds = torch.tensor(value_preds).to(**self.tpdv)
        returns = torch.tensor(returns).to(**self.tpdv_vn)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)
        returns_norm = returns.to(**self.tpdv)

        # Policy loss — lengths recomputed from EOA inside evaluate_actions
        logprob, entropy, values = self.policy.evaluate_actions(obs, padded_tokens)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # Value loss (clipped)
        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)
        value_clip_indicator = (value_pred_clipped - value_preds).abs() > self.ppo_clip
        value_clip_ratio = value_clip_indicator.to(**self.tpdv).mean()
        value_loss = value_loss.mean()

        entropy_loss = entropy.mean()

        # BC loss (optional)
        bc_loss = 0.0
        if demo_batch is not None:
            obs_demo_image, instruct_demo, padded_demo = demo_batch
            obs_demo = dict(image=torch.tensor(obs_demo_image).to(self.tpdv["device"]), task_description=instruct_demo)
            padded_demo = torch.tensor(padded_demo).to(self.tpdv["device"]).long()
            logprob_demo, _, _ = self.policy.evaluate_actions(obs_demo, padded_demo)
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
            bc_loss = bc_loss.item() if torch.is_tensor(bc_loss) else bc_loss
        log_str = (
            f"[UniVLA-Emu3 PPO Step {idx}/{total}] "
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

        return dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            bc_loss=bc_loss,
            grad_norm=grad_norm.item() if grad_norm is not None else 0.0,
            value_clip_ratio=value_clip_ratio.item(),
            ratio_mean=ratio.mean().item(),
        )
