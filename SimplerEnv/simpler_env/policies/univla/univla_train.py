"""
UniVLAPolicy — drop-in replacement for OpenVLAPolicy in the CRONOS RL loop.

Implements the exact same interface as OpenVLAPolicy:
  get_action, evaluate_actions, prep_rollout, prep_training, save, load
  + attributes: params_vla, params_vh, vla_optimizer, vh_optimizer, tpdv

Key differences from OpenVLA:
  - Action tokens: 4 VQ tokens (IDs > 32000) instead of 7 bin tokens
  - Continuous decode: stateful per-env ActionDecoder (CPU) from hidden states
  - Buffer stores latent_ids [B, 4] int32; env.step receives _last_cont_action [B, 7]
  - Center crop (90% area) applied before processor to match training distribution
"""

import json
import math
import sys
import os
from pathlib import Path
from PIL import Image as PILImage

import numpy as np
import torch
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
)

# Resolve paths so imports pick up the UniVLA subtree first.
# cronos_univla/UniVLA must be on sys.path for 'prismatic' to resolve to
# the UniVLA version (not openvla's copy).
_THIS_DIR = Path(__file__).resolve().parent
_UNIVLA_ROOT = _THIS_DIR.parents[3] / "UniVLA"  # cronos_univla/UniVLA
if str(_UNIVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_UNIVLA_ROOT))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    UniVLAForActionPredictionWithValueHead,
)
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)

# ActionDecoder lives at the UniVLA root (not inside the prismatic subtree)
from univla_action_decoder import ActionDecoder  # noqa: E402


class UniVLAPolicy:
    """CRONOS policy wrapper around UniVLAForActionPredictionWithValueHead.

    Args:
        all_args: Parsed Args dataclass (from main.py).  Must contain at minimum:
            vla_path, vla_load_path, vla_unnorm_key, vla_lora_rank, vla_lr,
            vla_vhlr, vla_optim_beta1, vla_optim_beta2, vla_temperature,
            vla_temperature_eval, seed, univla_window_size, univla_decoder_path.
        device_id (int): CUDA device index for the VLA model.
        num_envs (int): Number of parallel environments (= number of ActionDecoder instances).
    """

    def __init__(self, all_args, device_id: int, num_envs: int):
        self.args = all_args
        self.device_id = device_id
        self.num_envs = num_envs

        self.tpdv = {"device": torch.device(f"cuda:{device_id}"), "dtype": torch.bfloat16}
        self.tpdv_vn = {"device": torch.device(f"cuda:{device_id}"), "dtype": torch.float32}

        # ------------------------------------------------------------------ #
        # HF auto-class registration                                           #
        # ------------------------------------------------------------------ #
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, UniVLAForActionPredictionWithValueHead)

        # ------------------------------------------------------------------ #
        # Processor / tokenizer                                                #
        # ------------------------------------------------------------------ #
        self.image_processor = PrismaticImageProcessor.from_pretrained(
            self.args.vla_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.vla_path, trust_remote_code=True, padding_side="left"
        )
        self.processor = PrismaticProcessor.from_pretrained(
            self.args.vla_path,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
        )

        # ------------------------------------------------------------------ #
        # VLA backbone                                                         #
        # ------------------------------------------------------------------ #
        self.vla = UniVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation=None,       # fallback to SDPA/Eager
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=f"cuda:{device_id}",
            vh_mode="a0",
        )

        # Seeded value head re-init (identical to OpenVLAPolicy)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        self.vla.value_head._init_weights()

        # ------------------------------------------------------------------ #
        # ActionDecoder: one stateful instance per environment, on CPU        #
        # ------------------------------------------------------------------ #
        decoder_path = (
            self.args.univla_decoder_path
            if self.args.univla_decoder_path
            else str(Path(self.args.vla_path) / "action_decoder.pt")
        )
        decoder_sd = torch.load(decoder_path, map_location="cpu")

        # Validate window_size against checkpoint.
        # Must match only the top-level "proj.0.weight" key (shape [7*window_size, hidden_dim]).
        # Using "proj" in k would also match MAPBlock internals like
        # "latent_action_pool.projection.weight" (shape [512, 4096], 512//7=73).
        proj_key = next(
            k for k in decoder_sd if k.startswith("proj.") and len(decoder_sd[k].shape) == 2
        )
        actual_ws = decoder_sd[proj_key].shape[0] // 7
        if actual_ws != self.args.univla_window_size:
            raise ValueError(
                f"action_decoder.pt window_size={actual_ws} but "
                f"args.univla_window_size={self.args.univla_window_size}. "
                f"Set --univla_window_size {actual_ws}."
            )

        self.action_decoders = []
        for _ in range(num_envs):
            dec = ActionDecoder(window_size=self.args.univla_window_size)
            dec.net.load_state_dict(decoder_sd)
            dec.eval()          # frozen at all times — never train()
            self.action_decoders.append(dec)

        # ------------------------------------------------------------------ #
        # LoRA                                                                 #
        # ------------------------------------------------------------------ #
        if not self.args.vla_load_path:
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=min(self.args.vla_lora_rank, 16),
                lora_dropout=0.0,
                target_modules=[
                    "proj", "qkv", "fc1", "fc2",           # vision
                    "q", "kv", "fc3",                       # projector
                    "q_proj", "k_proj", "v_proj", "o_proj", # LLM
                    "gate_proj", "up_proj", "down_proj", "lm_head",
                ],
                init_lora_weights="gaussian",
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            self.vla = PeftModel.from_pretrained(
                self.vla, self.args.vla_load_path, is_trainable=True
            )
            print(f"VLA load: {self.args.vla_load_path}")

            if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
                path = Path(self.args.vla_load_path) / "dataset_statistics.json"
                ds = json.load(open(path, "r"))
                self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = (
                    ds[self.args.vla_unnorm_key]
                )

        # Ensure value head remains trainable after PEFT wrapping
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        # Cast value head to float32 (tpdv_vn dtype) for training stability.
        # The backbone is loaded in bfloat16 but predict_latent_action_batch /
        # evaluate_latent_action pass space_hidden.to(torch.float32) into the
        # value head — a dtype mismatch if the weights are still bfloat16.
        self.vla.value_head.to(torch.float32)

        self.vla.print_trainable_parameters()

        # ------------------------------------------------------------------ #
        # Optimizers                                                           #
        # ------------------------------------------------------------------ #
        self.params_vh = None
        self.params_vla = None
        self.vh_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        if self.args.vla_load_path:
            training_state_path = Path(self.args.vla_load_path) / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(
                    training_state_path, map_location=self.tpdv["device"]
                )
                if "vh" in training_state:
                    self.vla.value_head.load_state_dict(
                        training_state["vh"], assign=True
                    )
                else:
                    print("Warning: value_head state not found in training_state")

                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(training_state["vh_optimizer"])
                self.vla_optimizer.load_state_dict(training_state["vla_optimizer"])
                print(f"Optimizer load: {self.args.vla_load_path}")
            else:
                print(f"Warning: training_state not found in {training_state_path}")

        # Cache for continuous action (populated in get_action, consumed in get_cont_action)
        self._last_cont_action: torch.Tensor = None

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _setup_optimizer(self):
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        self.params_vh = [
            p for n, p in self.vla.named_parameters()
            if "value_head" in n and p.requires_grad
        ]
        self.params_vla = [
            p for n, p in self.vla.named_parameters()
            if "value_head" not in n and p.requires_grad
        ]
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)

    def _preprocess_obs(self, x: dict, action: torch.Tensor = None) -> BatchFeature:
        """Preprocess observations (and optional latent action IDs) into model inputs.

        Args:
            x: dict with keys "image" (uint8 [B, H, W, C]) and "task_description" (list[str]).
            action: optional [B, 4] int64 latent VQ token IDs. When provided, builds the
                    full sequence including action tokens for PPO evaluation.

        Returns:
            BatchFeature ready for the model (on self.tpdv device).
        """
        images = x["image"]             # [B, H, W, C] uint8
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor)
        assert images.ndim == 4
        assert images.shape[3] == 3
        assert images.dtype == torch.uint8
        assert isinstance(task_description, list)
        assert isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)

        # ---- Center crop (90%-area) BEFORE permute, matching training distribution ----
        # Decision from Phase 0.5: simpler-bridge checkpoint trained with crop_scale=0.9.
        H, W = images.shape[1], images.shape[2]
        crop_h = int(H * math.sqrt(0.9))    # ≈ 455 for H=480
        crop_w = int(W * math.sqrt(0.9))    # ≈ 607 for W=640
        top  = (H - crop_h) // 2
        left = (W - crop_w) // 2
        images = images[:, top:top + crop_h, left:left + crop_w, :]   # [B, crop_h, crop_w, C]

        # Convert to a list of PIL Images for PrismaticImageProcessor.
        # The processor expects PIL Images (it calls img.convert("RGB") internally)
        # and handles its own resize/normalise transforms; passing a tensor crashes with
        # AttributeError: 'Tensor' object has no attribute 'convert'.
        pil_images = [PILImage.fromarray(images[i].cpu().numpy()) for i in range(images.shape[0])]

        # ---- Build text prompts ----
        if action is None:
            task_prompt = [
                f"In: What action should the robot take to {t.lower()}?\nOut: "
                for t in task_description
            ]
        else:
            assert isinstance(action, torch.Tensor)
            # skip_special_tokens=False so VQ special tokens (<ACT_N>) are preserved as strings
            action_str = self.tokenizer.batch_decode(
                action.cpu(), skip_special_tokens=False
            )
            task_prompt = [
                f"In: What action should the robot take to {t.lower()}?\nOut: {a}</s>"
                for t, a in zip(task_description, action_str)
            ]

        inputs = self.processor(task_prompt, pil_images, padding=True)
        inputs = inputs.to(**self.tpdv)

        if action is not None:
            inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    # ---------------------------------------------------------------------- #
    # Public interface (matches OpenVLAPolicy exactly)                        #
    # ---------------------------------------------------------------------- #

    def get_action(
        self, x: dict, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout inference step.

        Returns:
            values   [B, 1] float32 — value estimates from value head
            latent_ids [B, 4] int64 — VQ token IDs for buffer storage
            logprobs [B, 1] float32 — log π(latent_ids | obs)

        Side effect:
            Populates self._last_cont_action [B, 7] float32 with the
            continuous 7D actions decoded by ActionDecoder (one per env).
            Must be retrieved via get_cont_action() before the next call.
        """
        temperature = (
            self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        )
        do_sample = temperature != 0.0
        features = self._preprocess_obs(x)

        values, latent_ids, logprobs, latent_action, visual_embed = (
            self.vla.predict_latent_action_batch(
                **features,
                unnorm_key=self.args.vla_unnorm_key,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.9,
            )
        )

        assert values.ndim == 2 and values.shape[1] == 1
        assert latent_ids.ndim == 2 and latent_ids.shape[1] == 4
        assert logprobs.ndim == 2 and logprobs.shape[1] == 1

        # Decode continuous 7D actions per env using stateful ActionDecoder (on CPU)
        stats = self.vla.get_action_stats(self.args.vla_unnorm_key)
        mask        = np.array(stats.get("mask", np.ones(7, dtype=bool)))
        action_low  = np.array(stats["q01"])
        action_high = np.array(stats["q99"])

        B = latent_ids.shape[0]
        cont_list = []
        for i in range(B):
            la_i  = latent_action[i:i + 1].cpu()   # [1, 4, hidden]
            ve_i  = visual_embed[i:i + 1].cpu()     # [1, 256, hidden]
            cont  = self.action_decoders[i](la_i, ve_i, mask, action_low, action_high)
            cont_list.append(cont)

        self._last_cont_action = torch.tensor(
            np.stack(cont_list, axis=0), dtype=torch.float32
        )   # [B, 7]

        return values, latent_ids, logprobs

    def get_cont_action(self) -> torch.Tensor:
        """Return the continuous 7D actions computed in the most recent get_action() call.

        Must be called immediately after get_action() before the next call
        overwrites the cache.

        Returns:
            [B, 7] float32 continuous actions (world_vector + rotation_delta + gripper).
        """
        return self._last_cont_action

    def evaluate_actions(
        self, x: dict, latent_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO training forward pass (with gradients).

        Args:
            x: observation dict (same format as get_action).
            latent_ids: [B, 4] int64 VQ token IDs stored in the replay buffer.

        Returns:
            logprobs [B, 1], entropy [B, 1], values [B, 1]
        """
        features = self._preprocess_obs(x, action=latent_ids)
        logprobs, entropy, values = self.vla.evaluate_latent_action(
            **features, unnorm_key=self.args.vla_unnorm_key
        )

        assert logprobs.shape == (latent_ids.shape[0], 1)
        assert entropy.shape  == (latent_ids.shape[0], 1)
        assert values.shape   == (latent_ids.shape[0], 1)

        return logprobs, entropy, values

    def reset_action_decoders(self):
        """Reset temporal ensemble buffers in all ActionDecoder instances.

        Must be called at task-switch boundaries (every task_len steps) so that
        action predictions from the previous task do not bleed into the new one.
        """
        for dec in self.action_decoders:
            dec.reset()

    def prep_rollout(self):
        """Switch VLA to eval mode for rollout (ActionDecoders stay in eval always)."""
        self.vla.eval()

    def prep_training(self):
        """Switch VLA to train mode for PPO updates."""
        self.vla.train()

    def save(self, path: Path, extra_state: dict = None):
        """Save LoRA adapter, value head, and optimizer states.

        ActionDecoder weights are NOT saved — they are pre-trained and frozen;
        the original action_decoder.pt at args.vla_path is re-loaded on resume.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.vla.save_pretrained(str(path))

        training_state = {
            "vh":            self.vla.value_head.state_dict(),
            "vh_optimizer":  self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        if extra_state:
            training_state.update(extra_state)
        torch.save(training_state, path / "training_state.pt")

        json.dump(
            self.vla.base_model.norm_stats,
            open(path / "dataset_statistics.json", "w"),
        )

    def load(self, path: Path):
        """Reload from a CRONOS checkpoint directory.

        Reinstantiates the VLA, applies the saved LoRA adapter, reloads the value
        head and optimizer states, and re-initialises ActionDecoder instances from
        the original pre-trained weights.
        """
        path = Path(path)

        del self.vla
        torch.cuda.empty_cache()

        self.vla = UniVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation=None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=f"cuda:{self.device_id}",
            vh_mode="a0",
        )
        self.vla = PeftModel.from_pretrained(self.vla, str(path), is_trainable=True)
        self.vla.print_trainable_parameters()

        # Ensure value head is trainable after PeftModel wrapping
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
            ds = json.load(open(path / "dataset_statistics.json", "r"))
            self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = (
                ds[self.args.vla_unnorm_key]
            )

        training_state = torch.load(
            path / "training_state.pt", map_location=self.tpdv["device"]
        )

        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state["vh"], assign=True)
        else:
            print("Warning: value_head state not found in training_state")

        # Keep value head in float32 after loading (weights may be bfloat16 on disk)
        self.vla.value_head.to(torch.float32)

        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(training_state["vh_optimizer"])
        self.vla_optimizer.load_state_dict(training_state["vla_optimizer"])

        # Re-instantiate ActionDecoders from the original frozen pre-trained weights
        decoder_path = (
            self.args.univla_decoder_path
            if self.args.univla_decoder_path
            else str(Path(self.args.vla_path) / "action_decoder.pt")
        )
        decoder_sd = torch.load(decoder_path, map_location="cpu")
        self.action_decoders = []
        for _ in range(self.num_envs):
            dec = ActionDecoder(window_size=self.args.univla_window_size)
            dec.net.load_state_dict(decoder_sd)
            dec.eval()
            self.action_decoders.append(dec)
