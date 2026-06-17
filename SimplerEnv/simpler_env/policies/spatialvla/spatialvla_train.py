"""SpatialVLA policy adapter for CRONOS — eval + PPO surface.

Mirrors `simpler_env.policies.openvla.openvla_train.OpenVLAPolicy` so CRONOS
can swap policies via `--policy` without touching the eval or PPO loops. PPO
surface (P-2 in `plans/2026-06-05_spatialvla-ppo-implementation.md`) mirrors
OpenVLA's openvla_train.py:21-271 — LoRA r32 on the L7 target set, two AdamW
(`params_vla` lr 1e-4 / `params_vh` lr 3e-3), full save/load round-trip.

Eval contract (called by `eval_only.py` and `wrapper.CronosWrapper`):
    - `__init__`         : load processor + model on GPU, set action-stats,
                            wrap in LoRA (zero-init B → no effect on zero-shot
                            forward), set up two optimizers
    - `_preprocess_obs`  : (image tensor, instruction list[, action_ids])
                            -> tokenized BatchFeature
    - `get_action`       : (obs, deterministic) -> (values, action_ids, logprobs)
    - `get_action_chunk` : K-step open-loop chunk; co-gates E-4 chunk(4)
    - `prep_rollout`     : `.eval()`
    - `get_action_stats` : delegates to the model (Option A' setter wired at load)

PPO contract (called by `CronosPPO.train_epoch` and the runner's PPO loop):
    - `evaluate_actions` : (obs, action_ids, return_diagnostics=False)
                            -> (logprobs, entropy, values[, diag])
    - `get_value`        : prompt-only value
    - `prep_training`    : `.train()`
    - `save` / `load`    : PEFT adapter + value head + both optimizers + norm_stats

Decisions baked in (eval plan §Approach):
    prompt   = f"What action should the robot take to {instruction}?"
    unnorm   = "bridge_orig/1.0.0"
    decoding = greedy when deterministic (top_k=0, top_p=1.0, temperature=0)

LoRA target_modules (the L7 set per the plan):
    Gemma2 (language_model): q_proj, k_proj, v_proj, o_proj,
                             gate_proj, up_proj, down_proj, lm_head
    SigLIP (vision_tower)  : out_proj, fc1, fc2
                             (q/k/v_proj overlap with Gemma2's names; peft
                              matches the suffix, so both backbones are hit)
    Projector              : linear  (only `multi_modal_projector.linear` ends
                              with `.linear`)
    Ego3D                  : position_embedding_head.0, position_embedding_head.3
    `modules_to_save=[]`     → `spatial_embed_tokens` stays frozen (M4 / NF-7
                              ground-truth: those embeddings are checkpoint-
                              specific to the sft-bridge bin geometry and must
                              not drift during PPO).
    `init_lora_weights="gaussian"` matches openvla_train.py:69.
"""
import copy
import json
from pathlib import Path
from typing import List, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoProcessor, BatchFeature

from model.modeling_spatialvla_valuehead import SpatialVLAForActionPredictionWithValueHead


# Prompt template — matches `Autonomous_RL/SpatialVLA/test/test_huggingface.py`
# and the eval plan §Approach ("Decisions baked in"). Lowercasing the
# instruction makes minor case differences ("Carrot" vs "carrot") inert.
_PROMPT_TEMPLATE = "What action should the robot take to {instruction}?"

# L7 LoRA target set (see module docstring for grouping / rationale).
_LORA_TARGET_MODULES = [
    # Gemma2 language model.
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",
    # SigLIP vision tower (q/k/v_proj also match Gemma2's; intentional).
    "out_proj", "fc1", "fc2",
    # Multi-modal projector (only `multi_modal_projector.linear` ends in `.linear`).
    "linear",
    # Ego3D position embedding head (Sequential's named children).
    "position_embedding_head.0", "position_embedding_head.3",
]


class SpatialVLAPolicy:
    """SpatialVLA-as-policy for CRONOS eval AND PPO.

    Argument shape mirrors `OpenVLAPolicy`: a single namespace `all_args`
    (an `Args` for training, an `EvalArgs` for eval) plus a device id.
    Reads `vla_path`, `vla_load_path`, `vla_unnorm_key`, `seed`,
    `vla_temperature(_eval)`, plus the PPO fields `vla_lora_rank`,
    `vla_lr`, `vla_vhlr`, `vla_optim_beta1`, `vla_optim_beta2`.
    """

    # NF-2 / H2: action token width per single action — buffer width. NOT the
    # continuous DoF (which is 7, returned by `get_action_dim` and consumed by
    # the wrapper decoder). main.py:333 reads this to size the replay buffer's
    # `actions.dat` memmap; conflating with DoF would either size the buffer
    # wrong or crash the per-step gather. See plan §P-3 / NF-2.
    act_token_len: int = 3

    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.tpdv = dict(device=torch.device(f"cuda:{device_id}"), dtype=torch.bfloat16)
        # `action_scale=1.0` matches `OpenVLAPolicy.__init__` (openvla_train.py:27).
        self.action_scale = 1.0

        # Processor — AutoProcessor dispatches to SpatialVLAProcessor via
        # processor_config.json's auto_map.
        #
        # `padding_side="left"` is REQUIRED, not cosmetic: with mixed-length
        # instructions in a batch, left-padding right-aligns every sequence so
        # (a) the prefill last-token hidden state used by the value head is at
        # index -1 for every sample, and (b) the first generated token aligns
        # across the batch. Right-padding would break both.
        self.processor = AutoProcessor.from_pretrained(
            self.args.vla_path, trust_remote_code=True,
        )
        self.processor.tokenizer.padding_side = "left"

        # Base model — raw checkpoint via `vla_path`. PEFT (if any) is layered
        # on top below; for the no-`vla_load_path` branch the LoRA B matrix is
        # zero-initialized so the forward at init is byte-identical to the raw
        # model (eval baseline unchanged).
        self.vla = SpatialVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=f"cuda:{device_id}",
            vh_mode="a0",
        )

        # Deterministic value-head init mirrors `OpenVLAPolicy.__init__:51-53`
        # (from_pretrained's internal HF init isn't seeded). Required for
        # P-1/P-4 gate determinism across reruns.
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        self.vla.value_head._init_weights()

        # Option A' wiring (E-2): bridge SpatialVLA's processor-side stats onto
        # the model so `policy.vla.get_action_stats(unnorm_key)` at
        # `eval_only.py:164` returns the bridge_orig/1.0.0 stats. Shared dict
        # reference, no copy, so the two views can't drift.
        self.vla.set_action_stats(self.processor.statistics)
        # NF-7 geometry derivation: single source of truth for the per-range
        # token bounds. Imported by the gates as module-level globals; calling
        # this BEFORE PEFT avoids any proxy-getattr surprises.
        self.vla.set_action_tokenizer(self.processor.action_tokenizer)

        # LoRA wrap (P-2 / M4). The no-`vla_load_path` branch is the fresh
        # PPO start; the load branch resumes a saved adapter. Either way the
        # value head is fully trainable (not LoRA-wrapped) and lives in a
        # separate optimizer (`vh_optimizer`).
        lora_rank = int(getattr(self.args, "vla_lora_rank", 32))
        if not getattr(self.args, "vla_load_path", ""):
            # Deterministic LoRA init: seed before get_peft_model so lora_A is
            # identical across reruns / parallel sessions.
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=min(lora_rank, 16),
                lora_dropout=0.0,
                target_modules=_LORA_TARGET_MODULES,
                # Freeze spatial_embed_tokens (M4): those embeddings are tied to
                # the checkpoint's specific bin geometry; modifying them via
                # `modules_to_save` would silently drift the action geometry
                # `set_action_tokenizer` derived at load.
                modules_to_save=[],
                init_lora_weights="gaussian",
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            self.vla = PeftModel.from_pretrained(
                self.vla, self.args.vla_load_path, is_trainable=True,
            )
            print(f"VLA load: {self.args.vla_load_path}")
            # Re-attach norm stats from sidecar if the loaded adapter's
            # processor stats disagree with the requested unnorm_key.
            stats_path = Path(self.args.vla_load_path) / "dataset_statistics.json"
            if stats_path.exists():
                ds = json.loads(stats_path.read_text())
                if self.args.vla_unnorm_key not in self.vla.base_model.model.statistics:
                    self.vla.base_model.model.statistics[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        # Re-enable the value head: `get_peft_model` freezes the entire base
        # model except LoRA params; we want value_head fully trainable (it
        # rides in `vh_optimizer`, not the LoRA-tracking `vla_optimizer`).
        # Mirrors openvla_train.py:82-84.
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        # Print the trainable-parameter breakdown (PEFT's own summary) once at
        # load — useful for sanity-checking the L7 target set actually matches.
        self.vla.print_trainable_parameters()

        # Optimizers — two AdamW per the lr split (1e-4 LoRA, 3e-3 value head).
        self.params_vh = None
        self.params_vla = None
        self.vh_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        # Resume optimizer state if the adapter came from disk.
        if getattr(self.args, "vla_load_path", ""):
            training_state_path = Path(self.args.vla_load_path) / "training_state.pt"
            if training_state_path.exists():
                ts = torch.load(training_state_path, map_location=self.tpdv["device"])
                if "vh" in ts:
                    # `assign=True` so dtype/device tensors are taken as-is
                    # rather than cast in place (mirrors openvla_train.py:101).
                    self.vla.value_head.load_state_dict(ts["vh"], assign=True)
                else:
                    print("Warning: value_head state not found in training_state")
                # Re-build optimizers AFTER weights are restored so Adam moments
                # snap to the right parameter tensors.
                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(ts["vh_optimizer"])
                self.vla_optimizer.load_state_dict(ts["vla_optimizer"])
                print(f"Optimizer load: {self.args.vla_load_path}")
            else:
                print(f"Warning: training_state not found in {training_state_path}")

    # --- optimizer split (M5 / NF-11) -----------------------------------------

    def _setup_optimizer(self):
        """Split trainable params into `params_vla` (LoRA) and `params_vh`
        (value head) and build the two AdamW optimizers."""
        self.params_vh = [
            p for n, p in self.vla.named_parameters()
            if "value_head" in n and p.requires_grad
        ]
        self.params_vla = [
            p for n, p in self.vla.named_parameters()
            if "value_head" not in n and p.requires_grad
        ]
        betas = (
            float(getattr(self.args, "vla_optim_beta1", 0.9)),
            float(getattr(self.args, "vla_optim_beta2", 0.999)),
        )
        self.vh_optimizer = AdamW(self.params_vh, lr=float(self.args.vla_vhlr), betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=float(self.args.vla_lr), betas=betas)

    # --- preprocessing ---------------------------------------------------------

    def _preprocess_obs(self, x: dict, action_ids: torch.Tensor = None) -> BatchFeature:
        """Tokenize (image, instruction[, action]) into the model's input layout.

        Inputs (CRONOS contract — same shape as OpenVLA's, see openvla_train.py:120):
            x["image"]            : torch.Tensor uint8 [B, H, W, 3]
            x["task_description"] : list[str] of length B
            action_ids (optional) : torch.LongTensor [B, ACTION_LEN]  — token ids
                                    of the sampled action (used only by the
                                    evaluate path; processor builds suffix +
                                    `token_type_ids` + `labels`).

        Returns a `BatchFeature` with keys:
            input_ids, attention_mask, pixel_values, intrinsic
            (+ token_type_ids, labels  when action_ids is provided).
        """
        images = x["image"]
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor)
        assert images.ndim == 4 and images.shape[3] == 3 and images.dtype == torch.uint8
        assert isinstance(task_description, list) and isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)

        # Convert torch tensor → list[np.ndarray] for SiglipImageProcessor.
        # Going through CPU/numpy is intentional: the processor does the
        # resize/normalize/permute itself, so it expects HWC-uint8 sources
        # (passing the GPU bf16 tensor confuses image-shape detection).
        images_np: List = [images[i].cpu().numpy() for i in range(images.shape[0])]

        prompts = [_PROMPT_TEMPLATE.format(instruction=t.lower()) for t in task_description]

        kwargs = dict(
            images=images_np,
            text=prompts,
            unnorm_key=self.args.vla_unnorm_key,
            return_tensors="pt",
            padding=True,
        )

        if action_ids is not None:
            # M3 / P-2: build the suffix from the EXACT sampled token ids.
            # `processor(..., suffix_actions=...)` would *re-discretize* via
            # the action tokenizer — wrong for off-policy scoring where we
            # need to score the actually-sampled tokens. Instead pass the
            # token strings directly via the `suffix=...` kwarg (consumed at
            # processing_spatialvla.py:122) — the processor still appends one
            # `eos` (L151) and emits `token_type_ids` + masked `labels` (L190).
            tok = self.processor.tokenizer
            suffix_strs = [
                "".join(tok.convert_ids_to_tokens(row.tolist()))
                for row in action_ids
            ]
            kwargs["suffix"] = suffix_strs

        inputs = self.processor(**kwargs)
        inputs = inputs.to(**self.tpdv)
        return inputs

    # --- inference (rollout) ---------------------------------------------------

    def get_action(self, x: dict, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step action (PPO unit + single-step eval baseline).

        NF-14 return-order trap: returns `(values, action_ids, logprobs)` —
        OPPOSITE order from `evaluate_actions`'s `(logprobs, entropy, values)`.
        Mirrors `OpenVLAPolicy.get_action` openvla_train.py:156-172.
        """
        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        do_sample = (temperature != 0.0)

        features = self._preprocess_obs(x)

        values, action_ids, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
        )

        assert values.ndim == 2 and values.shape[1] == 1
        assert action_ids.ndim == 2 and action_ids.shape[0] == values.shape[0]
        assert logprobs.ndim == 2 and logprobs.shape[1] == 1

        return values, action_ids, logprobs

    def get_action_chunk(self, x: dict, chunk: int) -> torch.Tensor:
        """K-step open-loop action chunk (E-4 chunk(4) co-gate; not a PPO unit).

        Returns `action_ids[B, 3*K]`.
        """
        features = self._preprocess_obs(x)
        action_ids = self.vla.predict_action_chunk(
            **features,
            chunk=chunk,
            do_sample=False,
        )
        assert action_ids.ndim == 2 and action_ids.shape[1] == 3 * chunk
        return action_ids

    # --- PPO surface (training) ------------------------------------------------

    def get_value(self, x: dict) -> torch.Tensor:
        """Prompt-only value query — used by gates (G3b) and any prompt-only
        PPO value lookup. Mirrors `OpenVLAPolicy.get_value`."""
        features = self._preprocess_obs(x)
        value = self.vla.get_value(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            pixel_values=features["pixel_values"],
            intrinsic=features["intrinsic"],
        )
        assert value.ndim == 2 and value.shape[1] == 1
        return value

    def evaluate_actions(self, x: dict, action_ids: torch.Tensor, return_diagnostics: bool = False):
        """Score (s, a) for the PPO update loop.

        NF-6: `return_diagnostics=False` for production callers so they unpack
        a 3-tuple; the gates pass `True` to grab the raw forward tensors and
        verify G2a/G3a with offsets written out independently.
        """
        features = self._preprocess_obs(x, action_ids=action_ids)
        out = self.vla.evaluate_action(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            pixel_values=features["pixel_values"],
            intrinsic=features["intrinsic"],
            labels=features["labels"],
            token_type_ids=features.get("token_type_ids"),
            unnorm_key=self.args.vla_unnorm_key,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            logprobs, entropy, values, diag = out
        else:
            logprobs, entropy, values = out
        assert logprobs.ndim == 2 and logprobs.shape[1] == 1
        assert entropy.ndim == 2 and entropy.shape[1] == 1
        assert values.ndim == 2 and values.shape[1] == 1
        return out

    # --- mode / stats ----------------------------------------------------------

    def prep_rollout(self) -> None:
        """Switch to eval mode. Matches `OpenVLAPolicy.prep_rollout`."""
        self.vla.eval()

    def prep_training(self) -> None:
        """Switch to train mode (enables dropout / disables eval-only paths).

        SpatialVLA's LoRA has dropout 0.0 and Gemma2 RMSNorm has no
        train/eval-dependent state, so this is mostly a no-op except for the
        attention-implementation dropout-rate gate (`config.training`).

        phaseO-5 history: gradient checkpointing on the LM trunk was attempted
        alongside `buffer_minibatch=40 / alg_gradient_accum=4` (plan §A3.1).
        Parity held (grad max|Δ| 1.75 ≤ TOL_GRAD 9.375) but PPO update was
        0.72× SLOWER — checkpointing recompute tax outweighed the per-minibatch
        overhead savings. Reverted: no checkpointing here, no minibatch
        widening in main.py."""
        self.vla.train()

    def get_action_stats(self, unnorm_key=None):
        """Pass-through to the model; the adapter wired Option A' at load."""
        return self.vla.get_action_stats(unnorm_key)

    # --- save / load (M5 / FR-9 / NF-11) ---------------------------------------

    def save(self, path: Path, extra_state: dict = None) -> None:
        """Persist adapter + value head + both optimizers + norm stats.

        Mirrors `OpenVLAPolicy.save` (openvla_train.py:226-239). The
        SpatialVLA-specific bit is the `norm_stats` dump: stats live on the
        processor (Option A'), so we serialize `self.vla.statistics` (which
        is the same dict object) to `dataset_statistics.json` for `load` to
        re-bridge after a fresh `from_pretrained`.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # PEFT adapter (adapter_config.json + adapter_*.safetensors).
        self.vla.save_pretrained(str(path))

        # Value head + optimizers + caller-supplied resume bookkeeping.
        training_state = {
            "vh": self.vla.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        if extra_state:
            training_state.update(extra_state)
        torch.save(training_state, path / "training_state.pt")

        # norm_stats sidecar (Option A'). PeftModel.statistics proxies to the
        # underlying model; getattr walks through to find it.
        stats = self._statistics_dict()
        with open(path / "dataset_statistics.json", "w") as f:
            json.dump(stats, f)

    def load(self, path: Path) -> None:
        """Reload the full PPO surface from `path`. Mirrors `OpenVLAPolicy.load`.

        Tears down the current model, fresh-loads the base from `vla_path`,
        wraps it with `PeftModel.from_pretrained`, re-bridges stats + the
        action tokenizer (NF-7 geometry derivation), restores the value head
        and both optimizers from `training_state.pt`.
        """
        path = Path(path)

        del self.vla
        torch.cuda.empty_cache()

        self.vla = SpatialVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=f"cuda:{self.device_id}",
            vh_mode="a0",
        )
        # Re-bridge Option A' BEFORE PEFT-wrapping (so the model.statistics dict
        # is the processor's; otherwise PEFT's proxy may copy the attribute).
        self.vla.set_action_stats(self.processor.statistics)
        self.vla.set_action_tokenizer(self.processor.action_tokenizer)

        # Wrap with the saved PEFT adapter (trainable so resumed training can
        # take optimizer steps).
        self.vla = PeftModel.from_pretrained(self.vla, str(path), is_trainable=True)
        self.vla.print_trainable_parameters()

        # If the saved adapter had a custom norm_stats sidecar (e.g., trained
        # on a different unnorm_key), merge it back in.
        stats_path = path / "dataset_statistics.json"
        if stats_path.exists():
            ds = json.loads(stats_path.read_text())
            stats = self._statistics_dict()
            for k, v in ds.items():
                # Don't blindly overwrite the processor's pinned stats — only
                # add entries the live dict is missing (e.g., the user trained
                # under a different unnorm_key).
                if k not in stats:
                    stats[k] = v

        training_state_path = path / "training_state.pt"
        ts = torch.load(training_state_path, map_location=self.tpdv["device"])

        if "vh" in ts:
            self.vla.value_head.load_state_dict(ts["vh"], assign=True)
        else:
            print("Warning: value_head state not found in training_state")

        # Re-enable value_head trainable + rebuild optimizers BEFORE loading
        # their state, so the param ordering matches what was saved.
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True
        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(ts["vh_optimizer"])
        self.vla_optimizer.load_state_dict(ts["vla_optimizer"])

    def _statistics_dict(self) -> dict:
        """Return the model's statistics dict (proxies through PEFT if wrapped)."""
        # PeftModel proxies via __getattr__ → base_model.model → our model.
        # `statistics` is a regular instance attribute; getattr finds it.
        return self.vla.statistics
