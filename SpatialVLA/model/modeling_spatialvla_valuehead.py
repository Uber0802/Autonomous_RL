# coding=utf-8
"""
SpatialVLA actor + value head (eval + PPO surface).

Mirrors `prismatic.extern.hf.modeling_prismatic.OpenVLAForActionPredictionWithValueHead`
(`Autonomous_RL/openvla/prismatic/extern/hf/modeling_prismatic.py:627`) so the CRONOS
PPO seam can swap SpatialVLA in for OpenVLA transparently.

Eval surface (E-1, unchanged):
    - `predict_action_batch`     single-step constrained generation (PPO unit)
    - `predict_action_chunk`     chunk(K) baseline (co-gates E-4 vs the paper)
    - `set_action_stats` / `get_action_stats` / `get_action_dim`

PPO surface (P-1, this file):
    - `evaluate_action(..., return_diagnostics=False)`
        per-range logπ + entropy + value at the suffixed sequence's last prompt token;
        `return_diagnostics=True` adds a 4th element with the raw forward tensors
        (NF-8 ground-truth anchor for G3a; NF-7 derived geometry; NF-12 intrinsic asserted).
    - `get_value`                value head on the prompt-only prefill last-token hidden
    - geometry derivation        `set_action_tokenizer` writes RANGES from the live
                                 SpatialActionTokenizer (NF-7, single source of truth)
    - `_last_gen_kwargs`         recorded each `predict_action_batch` call (G2d / NF-5)

Stats source — Option A' (decision 2026-06-06):
    SpatialVLA keeps unnorm stats on the *processor*, not the model: the
    checkpoint has no `dataset_statistics.json` and the model config carries
    no `norm_stats`. To honor the OpenVLA contract that `eval_only.py:164`
    reads stats from the MODEL, the adapter (E-2) calls
    `model.set_action_stats(processor.statistics)` once at load. The model
    and processor share the same dict, so they can't drift.
"""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from .modeling_spatialvla import SpatialVLAForConditionalGeneration


# --- Action-token geometry ----------------------------------------------------
# Single source of truth (NF-7): the model class DERIVES the per-range bounds at
# load time from `config.action_token_begin_idx` + the live `SpatialActionTokenizer`
# sub-tokenizer boundaries, then writes them into the module-level globals below.
# Gate code imports these globals (`from modeling_spatialvla_valuehead import RANGES, ...`)
# so it never re-hardcodes the literals — a checkpoint with a different `num_bins`
# or `action_token_begin_idx` (e.g. mix-pt vs sft-bridge) cannot silently corrupt
# every per-range gather.
#
# The literals below are the *expected* sft-bridge values and serve as a tripwire:
# `gate_geometry` asserts derived == expected, so a checkpoint with different geometry
# fails loudly before any per-range gather runs.
ACTION_LEN = 3                  # tokens per single action: (translation, rotation, gripper)
EOS_ID = 1                      # Gemma2 EOS — not used as an action token

RANGES = (
    ("trans", 257153, 261249),  # translation = theta*phi*r = 16*32*8 = 4096
    ("rot",   261249, 265345),  # rotation    = roll*pitch*yaw = 16*16*16 = 4096
    ("grip",  265345, 265347),  # gripper     = 2
)
RANGE_LO = RANGES[0][1]         # 257153 — lowest action-token id (== action_token_begin_idx)
RANGE_HI = RANGES[-1][2]        # 265347 — exclusive upper bound

# Back-compat for the E-1 eval path (used by `PositionRangeProcessor`).
ACTION_RANGES = tuple((lo, hi) for _, lo, hi in RANGES)


class ValueHead(nn.Module):
    """3-layer MLP value head.

    Copied from OpenVLA's eval-side value head
    (`Autonomous_RL/openvla/prismatic/extern/hf/modeling_prismatic.py:602`) so
    that PPO can use one identical recipe across policies — only the input
    dimensionality differs (SpatialVLA's Gemma2-2B hidden_size=2304 vs
    OpenVLA's Llama-2-7B hidden_size=4096).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.head_l1 = nn.Linear(hidden_size, 512)
        self.head_act1 = nn.GELU()
        self.head_l2 = nn.Linear(512, 128)
        self.head_act2 = nn.GELU()
        self.head_l3 = nn.Linear(128, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.head_l1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.head_l1.bias)
        nn.init.kaiming_normal_(self.head_l2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.head_l2.bias)
        nn.init.normal_(self.head_l3.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head_act1(self.head_l1(x))
        x = self.head_act2(self.head_l2(x))
        x = self.head_l3(x)
        return x


class PositionRangeProcessor(LogitsProcessor):
    """Mask logits at decode step *t* to the sub-range for position `(t mod 3)`.

    Generalizes to chunk-of-N actions: `predict_action_chunk(chunk=K)` generates
    `3*K` tokens, and at each step *t* the position-cycle `t mod 3` still selects
    the correct sub-range (translation, rotation, gripper).

    `prompt_len` is the input prefix length so we can compute `t` from the
    `input_ids` passed in by HF generate (which carries prompt + tokens
    generated so far).
    """

    def __init__(self, prompt_len: int):
        super().__init__()
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.prompt_len
        lo, hi = ACTION_RANGES[step % ACTION_LEN]
        # Mask everything outside [lo, hi); LogitsProcessor receives float scores,
        # so -inf is fine (softmax → 0 mass).
        scores[..., :lo] = -float("inf")
        scores[..., hi:] = -float("inf")
        return scores


def _maskaware_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """PaliGemma2 mask-aware, 1-indexed position_ids.

    HF `generate` computes mask-aware position_ids `cumsum-1` in
    `prepare_inputs_for_generation`, then SpatialVLA's wrapper
    (`modeling_spatialvla.py:474`) adds +1 → net `cumsum`.
    Pads are filled to 1 (HF convention; their attention is masked anyway).

    Cacheless `super().forward()` calls do NOT route through `prepare_inputs_for_generation`
    and default to `cache_position+1 = arange+1` (modeling_spatialvla.py:371-372) — a
    *non-mask-aware* phase that shifts left-padded prompts. G3 (rollout-prefill ↔
    evaluate-suffixed) compares two sequences of different total length but the same
    real-prompt length; with non-mask-aware positions the prompt tokens land at
    different rotary phases between the two and G3 spuriously fails. So evaluate
    paths must compute these themselves and pass them in.
    """
    pos = attention_mask.long().cumsum(-1)             # 1-indexed; pads → 0
    pos = pos.masked_fill(attention_mask == 0, 1)      # match HF: pads → position 1
    return pos


class SpatialVLAForActionPredictionWithValueHead(SpatialVLAForConditionalGeneration):
    """SpatialVLA + value head for CRONOS eval + PPO.

    Subclasses `SpatialVLAForConditionalGeneration` so all of SpatialVLA's
    image / Ego3D / Gemma2 forward+generate logic is inherited unchanged.
    """

    def __init__(self, config, vh_mode: str = "a0"):
        super().__init__(config)

        # SpatialVLA's value-head MVP only supports the prefill anchor (a0).
        # OpenVLA carries a/a6 variants that read other anchors — irrelevant
        # here; if someone needs them, mirror `OpenVLAForActionPredictionWithValueHead.__init__`.
        if vh_mode != "a0":
            raise ValueError(
                f"Unsupported vh_mode='{vh_mode}'. SpatialVLA wires only 'a0' "
                "(prefill last-prompt-token anchor)."
            )
        self.vh_mode = vh_mode
        self.value_head = ValueHead(config.text_config.hidden_size)

        # Stats are populated by the adapter via `set_action_stats` (Option A').
        # Kept `None` until set so a missed plumbing step fails loudly rather
        # than returning an empty dict and silently unnorming with zeros.
        self.statistics: Optional[Dict[str, Any]] = None

        # NF-7: action-token geometry is derived from the live SpatialActionTokenizer
        # at load (see `set_action_tokenizer`). Kept None until set so a missed
        # plumbing step fails loudly in evaluate_action's range checks.
        self.action_tokenizer = None

        # G2d / NF-5: every `predict_action_batch` call records the *intended*
        # generation kwargs so gates can assert (i) sampling was non-truncating
        # (top_k=0, top_p=1.0, temperature=1.0) and (ii) greedy was *configured*
        # greedy (do_sample=False) — temperature scaling is argmax-invariant, so
        # the output-side argmax check is blind to a stray temperature.
        self._last_gen_kwargs: Optional[Dict[str, Any]] = None

    # --- stats plumbing (Option A') --------------------------------------------

    def set_action_stats(self, statistics: Dict[str, Any]) -> None:
        """Bridge SpatialVLA's processor-side stats onto the model.

        `eval_only.py:164` reads norm stats from the MODEL (the OpenVLA path),
        but SpatialVLA's stats live on the processor (`processor.statistics`
        from `processor_config.json`). The adapter calls this once at load
        time so the model exposes the same dict — same object, no copy, so
        the two views can't drift.
        """
        self.statistics = statistics

    def set_action_tokenizer(self, action_tokenizer) -> None:
        """NF-7: install the live `SpatialActionTokenizer` and derive geometry from it.

        Single source of truth for the per-range token bounds. The adapter
        calls this once at load with `processor.action_tokenizer`; the test
        shim does the same with a hand-built tokenizer. Rewriting the
        module-level globals (rather than only instance attrs) is deliberate
        — gate code imports them as `from modeling_spatialvla_valuehead import RANGES, ...`
        per the plan, so they MUST be the values derived from the loaded
        checkpoint, not the file-level defaults.
        """
        global RANGES, RANGE_LO, RANGE_HI, ACTION_RANGES
        derived = (
            ("trans", action_tokenizer.translation_tokenizer.token_start_idx,
                       action_tokenizer.rotation_tokenizer.token_start_idx),
            ("rot",   action_tokenizer.rotation_tokenizer.token_start_idx,
                       action_tokenizer.gripper_tokenizer.token_start_idx),
            ("grip",  action_tokenizer.gripper_tokenizer.token_start_idx,
                       action_tokenizer.gripper_tokenizer.token_end_idx + 1),
        )
        # Range_LO must match config.action_token_begin_idx; checked here so a
        # config/tokenizer mismatch surfaces immediately instead of corrupting
        # every per-range gather downstream.
        assert derived[0][1] == self.config.action_token_begin_idx, (
            f"NF-7: action_tokenizer.translation_tokenizer.token_start_idx={derived[0][1]} "
            f"!= config.action_token_begin_idx={self.config.action_token_begin_idx}"
        )
        RANGES = derived
        RANGE_LO = derived[0][1]
        RANGE_HI = derived[-1][2]
        ACTION_RANGES = tuple((lo, hi) for _, lo, hi in derived)
        self.action_tokenizer = action_tokenizer

    def _check_unnorm_key(self, unnorm_key: Optional[str]) -> str:
        """Resolve `unnorm_key`, raising clear errors for the two failure modes."""
        if self.statistics is None:
            raise RuntimeError(
                "SpatialVLAForActionPredictionWithValueHead.statistics is unset. "
                "The adapter must call `set_action_stats(processor.statistics)` "
                "at load time (Option A')."
            )
        if unnorm_key is None:
            if len(self.statistics) != 1:
                raise AssertionError(
                    "Model carries more than one dataset's statistics; pass "
                    f"`unnorm_key` explicitly. Available: {list(self.statistics.keys())}"
                )
            return next(iter(self.statistics.keys()))
        if unnorm_key not in self.statistics:
            raise AssertionError(
                f"`unnorm_key={unnorm_key!r}` not in statistics. "
                f"Available: {list(self.statistics.keys())}"
            )
        return unnorm_key

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Return `{q01, q99, mask}` for the named dataset, mirroring OpenVLA."""
        key = self._check_unnorm_key(unnorm_key)
        return self.statistics[key]["action"]

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Action vector dimensionality (7 for WidowX EE-delta)."""
        key = self._check_unnorm_key(unnorm_key)
        return len(self.statistics[key]["action"]["q01"])

    # --- eval-time inference ---------------------------------------------------

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        intrinsic: torch.Tensor,
        unnorm_key: Optional[str] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        """Batched constrained single-step action generation (PPO unit).

        Mirrors `OpenVLAForActionPredictionWithValueHead.predict_action_batch`
        (`openvla/.../modeling_prismatic.py:777`). Runs `generate(max_new_tokens=3)`
        with a per-step range mask, gathers the per-range log-softmax of the chosen
        token at each of the 3 positions, sums to one scalar, and reads the value
        from the prefill last-prompt-token hidden state (vh_mode='a0').

        NF-14 return-order trap: returns `(values, action_ids, logprobs)` —
        OPPOSITE order from `evaluate_action`'s `(logprobs, entropy, values)`.
        Mirrors OpenVLA `modeling_prismatic.py:839` vs `:718`; do not "normalize."

        Returns:
            values     [B, 1]  — value head on prefill last-token hidden
            action_ids [B, 3]  — generated action token ids (trans, rot, grip)
            logprobs   [B, 1]  — Σ_t log_softmax(logits[t][lo:hi])[id_t - lo]
        """
        self._check_unnorm_key(unnorm_key)  # validate stats are wired
        # NF-12: intrinsic is Optional in SpatialVLA's forward
        # (modeling_spatialvla.py:340), so a forgotten/None intrinsic silently
        # disables the 3D (Ego3D/ZoeDepth) path instead of erroring. Surface it.
        assert intrinsic is not None, "intrinsic is required (NF-12: forward Optional-typed it)"
        prompt_len = input_ids.shape[1]

        # G2d / NF-5: record the intent. A later edit that drops top_k=0 (HF
        # would default top_k=50 and truncate the 4096-way categorical) regresses
        # loudly via `gate_G2d_no_truncation`; the live HF behavior is independently
        # checked by `gate_G2d_warper_absent`.
        self._last_gen_kwargs = dict(
            do_sample=do_sample,
            temperature=(temperature if do_sample else 1.0),
            top_k=top_k,
            top_p=top_p,
        )

        output = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            intrinsic=intrinsic,
            max_new_tokens=ACTION_LEN,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k,
            top_p=top_p,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_logits=True,
            logits_processor=LogitsProcessorList([PositionRangeProcessor(prompt_len)]),
            **kwargs,
        )
        action_ids = output.sequences[:, -ACTION_LEN:]  # [B, 3]

        # G-shape: every token landed in its position's sub-range. If this trips
        # in production the PositionRangeProcessor wasn't applied — surface it.
        for t, (lo, hi) in enumerate(ACTION_RANGES):
            assert torch.all(action_ids[:, t] >= lo) and torch.all(action_ids[:, t] < hi), (
                f"action_ids[:, {t}] outside sub-range [{lo}, {hi})"
            )

        # Per-range log_softmax + gather, summed across the 3 positions.
        # `output.logits` is a tuple of length 3, each `[B, vocab_size]`.
        # Slicing then log_softmax conditions on "the token is from the sub-range"
        # — which is exactly the actor distribution after the mask.
        logprobs = action_ids.new_zeros((action_ids.shape[0], 1), dtype=torch.float32)
        for t, (lo, hi) in enumerate(ACTION_RANGES):
            logits_t = output.logits[t][:, lo:hi].float()        # [B, range_size]
            logp_t = F.log_softmax(logits_t, dim=-1)             # [B, range_size]
            idx_t = (action_ids[:, t] - lo).unsqueeze(-1)        # [B, 1]
            logprobs += torch.gather(logp_t, 1, idx_t)

        # Value head on prefill last-prompt-token hidden state (vh_mode='a0').
        # `output.hidden_states[0][-1]` is the prefill (generation step 0),
        # last decoder layer; `[:, -1]` is the last token of the (left-padded)
        # prompt, which is the same text position across the whole batch.
        prefill_last_layer = output.hidden_states[0][-1]         # [B, prompt_len, H]
        hidden_features = prefill_last_layer[:, -1]              # [B, H]
        values = self.value_head(hidden_features)                # [B, 1]

        return values, action_ids, logprobs

    @torch.no_grad()
    def predict_action_chunk(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        intrinsic: torch.Tensor,
        chunk: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **kwargs,
    ) -> torch.LongTensor:
        """Batched constrained chunk-of-K action generation (no logprob/value).

        Eval-only co-gate at E-4: matches the SpatialVLA paper's 4-action-chunk
        open-loop deployment (`action_chunk_size=4` in `processor_config.json`).
        Generates `3*K` action tokens; the position-range mask uses `t mod 3`,
        so each token still lands in the right sub-range.

        Returns:
            action_ids [B, 3*K]
        """
        prompt_len = input_ids.shape[1]
        output = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            intrinsic=intrinsic,
            max_new_tokens=ACTION_LEN * chunk,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k,
            top_p=top_p,
            return_dict_in_generate=True,
            logits_processor=LogitsProcessorList([PositionRangeProcessor(prompt_len)]),
            **kwargs,
        )
        action_ids = output.sequences[:, -ACTION_LEN * chunk:]  # [B, 3*K]
        return action_ids

    # --- PPO surface (P-1) -----------------------------------------------------

    def evaluate_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        intrinsic: torch.Tensor,
        labels: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        return_diagnostics: bool = False,
    ):
        """Score a (left-padded, suffixed) batch — used by `CronosPPO.train_epoch`.

        Mirrors `OpenVLAForActionPredictionWithValueHead.evaluate_action`
        (`openvla/.../modeling_prismatic.py:651`) but with SpatialVLA's 3-composite-token
        action and the per-range renormalized softmax that matches the constrained
        rollout sampler.

        Input is the **suffixed** sequence `[…prompt…][trans][rot][grip][eos]`
        (`processing_spatialvla.py:151,190`, verified 2026-06-06):
            input_ids[:, -1]                 = eos (id 1)
            input_ids[:, -ACTION_LEN-1:-1]   = (trans, rot, grip)
            input_ids[:, -ACTION_LEN-2]      = last prompt token (value anchor)
        Predictions for the action tokens come from the logits at positions
        [-ACTION_LEN-2 : -2] (next-token at position p predicts token at p+1).
        Value reads `hidden_states[-1][:, -ACTION_LEN-2]`, the SAME text position
        as `predict_action_batch`'s prefill `[:, -1]` — that's what G3 enforces.

        NF-12: `intrinsic` is required, not optional — SpatialVLA's forward
        types it `Optional` (modeling_spatialvla.py:340), so `None` silently
        disables the Ego3D 3D path instead of erroring.

        NF-14: returns `(logprobs, entropy, values)` — OPPOSITE order from
        `predict_action_batch`'s `(values, action_ids, logprobs)`. Mirrors
        OpenVLA `modeling_prismatic.py:718` vs `:839`; do not "normalize."

        Returns 3-tuple by default; with `return_diagnostics=True` returns a
        4-tuple where the 4th element is `{logits_slice, hidden_anchor,
        hidden_all, input_ids}` from the SAME raw forward, so a test (G2a,
        G3a) can recompute logπ/entropy AND the value anchor with the offset
        math written out independently — NF-7/NF-8 ground-truth checks. Default
        False so production callers (`ppo.py:38`, NF-6) still unpack a 3-tuple.
        """
        assert intrinsic is not None, "intrinsic is required (NF-12)"
        assert labels is not None, "labels are required to drive forward into training mode (prefix-LM mask)"
        assert token_type_ids is not None, "token_type_ids required (processor builds them when suffix is set)"
        if RANGE_LO is None or self.action_tokenizer is None:
            raise RuntimeError(
                "Action-token geometry not derived. The adapter must call "
                "`set_action_tokenizer(processor.action_tokenizer)` at load (NF-7)."
            )

        # Mask-aware, 1-indexed position_ids — match HF generate's prefill phase
        # (see `_maskaware_position_ids` docstring). Critical for G3.
        position_ids = _maskaware_position_ids(attention_mask)

        # G3 fix: SpatialVLA's processor labels pad columns with token_type_ids==0
        # (same type as the prompt). _update_causal_mask (modeling_spatialvla.py:304-305)
        # applies the prefix-LM `type==0 → 0` AFTER the padding-fill, so pad columns
        # get RE-UNMASKED at the suffixed forward — but at the rollout's prompt-only
        # prefill (is_training=False) the re-unmask step never runs, so pads stay
        # masked. That asymmetry shifts the last-prompt-token hidden state between
        # the two paths by O(0.1-0.3) and trips G3/G2b. Mark pads as type==1 here
        # so the prefix-fill skips them — pad-attention now matches rollout. Labels
        # for pad positions stay ignored: the processor already set them to -100
        # via the original `token_type_ids==0` mask (processing_spatialvla.py:190).
        if token_type_ids is not None:
            token_type_ids = token_type_ids.clone()
            token_type_ids = token_type_ids.masked_fill(attention_mask == 0, 1)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            intrinsic=intrinsic,
            labels=labels,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        # Forward returns logits as fp32 when labels are passed (see
        # modeling_spatialvla.py:416 `logits = logits.float()`); use as-is.
        logits = outputs.logits                                # [B, L, V]
        last_hidden = outputs.hidden_states[-1]                # [B, L, H]  (raw, untouched)

        # Per-range gather: predictions for the action tokens are at positions
        # [-ACTION_LEN-2 : -2]; targets are at [-ACTION_LEN-1 : -1] = (trans, rot, grip).
        # We do NOT use a full-vocab log_softmax — the rollout sampler is constrained
        # to each sub-range by `PositionRangeProcessor`, so the correct base measure
        # at scoring time is the per-range renormalized categorical (matches the
        # behaviour distribution exactly). For a trained policy the action-range
        # mass ≈ 1 so this is ≈ identity; G2a / G2b enforce internal consistency.
        logits_slice = logits[:, -ACTION_LEN - 2 : -2, :]      # [B, 3, V]
        action_ids = input_ids[:, -ACTION_LEN - 1 : -1]        # [B, 3]

        logprobs = action_ids.new_zeros((action_ids.shape[0], 1), dtype=torch.float32)
        ent_per_step = []
        for t, (_, lo, hi) in enumerate(RANGES):
            logits_t = logits_slice[:, t, lo:hi].float()       # [B, range_size]
            logp_t = F.log_softmax(logits_t, dim=-1)           # [B, range_size]
            idx_t = (action_ids[:, t] - lo).unsqueeze(-1)      # [B, 1]
            logprobs += torch.gather(logp_t, 1, idx_t)
            p_t = logp_t.exp()
            ent_per_step.append(-(p_t * logp_t).sum(dim=-1))   # [B]
        entropy = torch.stack(ent_per_step, dim=1).mean(dim=1, keepdim=True)  # [B, 1]

        # Value at the last prompt token (vh_mode='a0').
        hidden_anchor = last_hidden[:, -ACTION_LEN - 2]        # [B, H]
        values = self.value_head(hidden_anchor.to(self.value_head.head_l1.weight.dtype))

        if return_diagnostics:
            diag = {
                # Raw forward tensors — NOT massaged. The test recomputes from these
                # with offset math written out independently to catch a wrong gather
                # range (G2a) and a wrong layer / anchor offset (G3a, NF-8).
                "logits_slice": logits_slice,
                "hidden_anchor": hidden_anchor,
                "hidden_all": last_hidden,
                "input_ids": input_ids,
            }
            return logprobs, entropy, values, diag
        return logprobs, entropy, values

    def get_value(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        intrinsic: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        """Value head on the PROMPT-ONLY prefill last-token hidden state.

        Mirrors `OpenVLAForActionPredictionWithValueHead.get_value`
        (`openvla/.../modeling_prismatic.py:720`). Used by gates (G3b) and any
        prompt-only PPO value query; for the in-loop rollout, `predict_action_batch`
        already returns the prefill value, so callers typically don't need this.

        Same conventions as `evaluate_action`: NF-12 intrinsic required,
        mask-aware 1-indexed position_ids.
        """
        assert intrinsic is not None, "intrinsic is required (NF-12)"
        position_ids = _maskaware_position_ids(attention_mask)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            intrinsic=intrinsic,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]                # [B, L, H]
        hidden_features = last_hidden[:, -1]                   # [B, H]  (last prompt token)
        values = self.value_head(hidden_features.to(self.value_head.head_l1.weight.dtype))
        return values
