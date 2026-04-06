"""
Emu3MoE model adapted for RL (PPO) training with FAST action tokenization.

Adds ValueHead and RL-specific methods (predict_action_batch, evaluate_action,
get_value) to the base Emu3MoE model. Handles variable-length FAST BPE action
sequences that terminate with an EOA token.

Key differences from the OpenVLA/Prismatic path:
    - Actions are FAST BPE tokens (variable length, typically 20-50 tokens)
    - Each action chunk decodes to 10 timesteps × 7 dims via IDCT
    - Vocab id → FAST BPE id: bpe_id = last_vocab_idx - vocab_id
    - Log-prob aggregation sums over actual (non-padded) length per sample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from transformers import LogitsProcessor
from transformers.generation import LogitsProcessorList, GenerationConfig

import sys
import os

# Add Emu3 reference to path
_emu3_path = os.path.join(os.path.dirname(__file__), '..', 'reference', 'Emu3')
if _emu3_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_emu3_path))

from emu3.mllm.modeling_emu3 import Emu3MoE


class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    """Constrains generation to only produce action BPE tokens or the eoa token."""
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        scores[~mask] = -float("inf")
        return scores


class ValueHead(nn.Module):
    """3-layer MLP value head, identical architecture to OpenVLA's."""
    def __init__(self, hidden_size):
        super().__init__()
        self.head_l1 = nn.Linear(hidden_size, 512)
        self.head_act1 = nn.GELU()
        self.head_l2 = nn.Linear(512, 128)
        self.head_act2 = nn.GELU()
        self.head_l3 = nn.Linear(128, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.head_l1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.head_l1.bias)
        nn.init.kaiming_normal_(self.head_l2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.head_l2.bias)
        nn.init.normal_(self.head_l3.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.head_act1(self.head_l1(x))
        x = self.head_act2(self.head_l2(x))
        x = self.head_l3(x)
        return x


class Emu3MoEForRL(Emu3MoE):
    """
    Emu3MoE extended with a value head for PPO training.

    Uses FAST BPE action tokenization with variable-length sequences. Actions are
    generated until an EOA token or max_action_tokens is reached. Log-probs are
    computed over the actual (non-padded) length.

    Key methods:
        predict_action_batch: Generate FAST action tokens and return
            (values, padded_tokens, actual_lengths, logprobs).
        evaluate_action: Teacher-forced evaluation of stored action sequences.
        get_value: Value-only forward pass using prompt hidden states.

    The caller is responsible for FAST → continuous action decoding (IDCT path).
    """

    # Constants matching the pretrained checkpoint
    ACTION_DIM = 7
    ACTION_CHUNK_SIZE = 10

    def __init__(self, config, vh_mode: str = "a0"):
        super().__init__(config)
        self.value_head = ValueHead(config.hidden_size)
        self.vh_mode = vh_mode

        # Action token range (set by setup_action_tokens)
        self.last_vocab_idx: Optional[int] = None
        self.fast_vocab_size: Optional[int] = None
        self.eoa_token_id: int = 151845
        self.allowed_token_ids: Optional[List[int]] = None

    def setup_action_tokens(
        self,
        last_vocab_idx: int,
        fast_vocab_size: int,
        eoa_token_id: int = 151845,
    ) -> None:
        """Configure action token range based on FAST tokenizer.

        The mapping is: bpe_id = last_vocab_idx - vocab_id
        Valid FAST BPE ids: [0, fast_vocab_size - 1]
        Valid Emu3 vocab ids: [last_vocab_idx - fast_vocab_size + 1, last_vocab_idx]
        Plus the EOA token for terminating generation.
        """
        self.last_vocab_idx = last_vocab_idx
        self.fast_vocab_size = fast_vocab_size
        self.eoa_token_id = eoa_token_id
        action_start = last_vocab_idx - fast_vocab_size + 1
        action_end = last_vocab_idx + 1
        self.allowed_token_ids = list(range(action_start, action_end)) + [eoa_token_id]

    # ------------------------------------------------------------------
    # Helper: slice logits to FAST action vocab + EOA
    # ------------------------------------------------------------------
    def _action_logits_slice(self, logits: torch.Tensor) -> torch.Tensor:
        """Slice vocab logits to just the action BPE range + EOA.

        Args:
            logits: [..., vocab_size]
        Returns:
            sliced: [..., fast_vocab_size + 1] with EOA appended as last index.
        """
        action_start = self.last_vocab_idx - self.fast_vocab_size + 1
        action_end = self.last_vocab_idx + 1
        action_logits = logits[..., action_start:action_end]  # [..., fast_vocab_size]
        eoa_logits = logits[..., self.eoa_token_id:self.eoa_token_id + 1]  # [..., 1]
        return torch.cat([action_logits, eoa_logits], dim=-1)  # [..., fast_vocab_size + 1]

    def _vocab_ids_to_slice_ids(self, vocab_ids: torch.Tensor) -> torch.Tensor:
        """Map generated vocab ids to indices within the sliced action logit range.

        Args:
            vocab_ids: [...] Long tensor of vocab ids, each either in the
                action range or equal to eoa_token_id.
        Returns:
            slice_ids: [...] indices into _action_logits_slice output.
        """
        action_start = self.last_vocab_idx - self.fast_vocab_size + 1
        eoa_idx = self.fast_vocab_size  # Position of EOA in the sliced logits
        is_eoa = (vocab_ids == self.eoa_token_id)
        slice_ids = vocab_ids - action_start
        slice_ids = torch.where(is_eoa, torch.full_like(slice_ids, eoa_idx), slice_ids)
        return slice_ids.clamp(0, self.fast_vocab_size)

    # ------------------------------------------------------------------
    # Generation: predict actions + return values + logprobs
    # ------------------------------------------------------------------
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        max_action_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate FAST action tokens via constrained decoding.

        Args:
            input_ids: [B, seq_len] — prompt ending in the boa token.
            attention_mask: [B, seq_len]
            max_action_tokens: hard cap on generated tokens (safety for no-EOA).
            do_sample: sampling vs greedy.
            temperature: sampling temperature.

        Returns:
            values: [B, 1] value estimates at the final prompt position.
            padded_tokens: [B, max_action_tokens] padded (with eoa) token ids.
            actual_lengths: [B] int64 tensor of tokens before eoa.
            logprobs: [B, 1] sum of log-probs over non-padded tokens (eoa
                included — the eoa decision is part of the policy).
        """
        assert self.last_vocab_idx is not None, "Call setup_action_tokens() first"
        B = input_ids.shape[0]
        prompt_len = input_ids.shape[1]

        action_logits_processor = ActionIDConstraintLogitsProcessor(self.allowed_token_ids)

        gen_config = GenerationConfig(
            pad_token_id=self.config.pad_token_id,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.eoa_token_id,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
        )

        output = self.generate(
            input_ids,
            gen_config,
            attention_mask=attention_mask,
            max_new_tokens=max_action_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_logits=True,
            logits_processor=LogitsProcessorList([action_logits_processor]),
        )

        # output.sequences: [B, prompt_len + n_gen]
        full_generated = output.sequences[:, prompt_len:]  # [B, n_gen]
        n_gen = full_generated.shape[1]

        # Pad to max_action_tokens with EOA (safe because we treat EOA as "end")
        if n_gen < max_action_tokens:
            pad_width = max_action_tokens - n_gen
            pad = full_generated.new_full((B, pad_width), self.eoa_token_id)
            padded_tokens = torch.cat([full_generated, pad], dim=1)
        else:
            padded_tokens = full_generated[:, :max_action_tokens]

        # Determine actual lengths (tokens before first EOA, exclusive).
        # Each sample has its own length.
        actual_lengths = torch.full((B,), max_action_tokens, dtype=torch.long, device=padded_tokens.device)
        for b in range(B):
            eoa_positions = (padded_tokens[b] == self.eoa_token_id).nonzero(as_tuple=True)[0]
            if len(eoa_positions) > 0:
                actual_lengths[b] = eoa_positions[0].item()

        # ---- Log-prob aggregation ----
        # output.logits is a tuple of [B, vocab_size] tensors, one per step.
        n_logits = min(len(output.logits), max_action_tokens)
        step_logits = torch.stack(output.logits[:n_logits], dim=1)  # [B, n_logits, vocab]
        action_logits = self._action_logits_slice(step_logits)  # [B, n_logits, slice]
        step_logprobs = F.log_softmax(action_logits, dim=-1)  # [B, n_logits, slice]

        step_ids = padded_tokens[:, :n_logits]  # the tokens that were actually generated
        slice_ids = self._vocab_ids_to_slice_ids(step_ids)  # [B, n_logits]
        gathered = torch.gather(step_logprobs, 2, slice_ids.unsqueeze(-1)).squeeze(-1)  # [B, n_logits]

        # Mask: count steps up to AND INCLUDING first EOA (eoa decision is part of policy)
        step_idx = torch.arange(n_logits, device=gathered.device).unsqueeze(0).expand(B, -1)
        length_mask = (step_idx <= actual_lengths.unsqueeze(1).clamp(max=n_logits - 1)).float()
        logprobs = (gathered * length_mask).sum(dim=1, keepdim=True)  # [B, 1]

        # ---- Value head ----
        # Hidden states at first generation step correspond to the final prompt token.
        first_step_hidden = output.hidden_states[0][-1]  # [B, prompt_len, H]
        hidden_features = first_step_hidden[:, -1]  # [B, H]
        values = self.value_head(hidden_features)  # [B, 1]

        return values, padded_tokens, actual_lengths, logprobs

    # ------------------------------------------------------------------
    # Teacher-forced evaluation (for PPO policy update)
    # ------------------------------------------------------------------
    def evaluate_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        action_tokens: torch.LongTensor,
        action_lengths: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced evaluation of given action token sequences.

        Args:
            input_ids: [B, full_len] — prompt + padded action tokens (with EOA).
                The last `action_tokens.shape[1]` positions correspond to the
                generated tokens (including EOA padding).
            attention_mask: [B, full_len] — 1 for valid tokens (prompt + actual
                action tokens up to and including EOA), 0 for padding beyond EOA.
            action_tokens: [B, max_action_tokens] — padded action token ids.
            action_lengths: [B] — actual length of each action sequence (number
                of BPE tokens before EOA; the EOA itself is included in logprob
                aggregation to match generation).

        Returns:
            logprobs: [B, 1] sum over (length + 1) tokens.
            entropy:  [B, 1] mean entropy across evaluated positions.
            values:   [B, 1] at the final prompt position (same as predict).
        """
        assert self.last_vocab_idx is not None, "Call setup_action_tokens() first"
        B = input_ids.shape[0]
        max_A = action_tokens.shape[1]
        prompt_len = input_ids.shape[1] - max_A

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # logits[t] predicts input_ids[t+1]. To score action_tokens[:, i],
        # we need logits at position (prompt_len - 1 + i).
        # That corresponds to slicing logits[:, prompt_len - 1 : prompt_len - 1 + max_A].
        action_logit_slice = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + max_A]  # [B, max_A, vocab]
        action_logits = self._action_logits_slice(action_logit_slice)  # [B, max_A, slice]
        step_logprobs = F.log_softmax(action_logits, dim=-1)

        slice_ids = self._vocab_ids_to_slice_ids(action_tokens)  # [B, max_A]
        gathered = torch.gather(step_logprobs, 2, slice_ids.unsqueeze(-1)).squeeze(-1)  # [B, max_A]

        # Mask over (action_lengths + 1) positions (include EOA decision)
        step_idx = torch.arange(max_A, device=gathered.device).unsqueeze(0).expand(B, -1)
        length_mask = (step_idx <= action_lengths.unsqueeze(1).clamp(max=max_A - 1)).float()
        logprobs = (gathered * length_mask).sum(dim=1, keepdim=True)  # [B, 1]

        # Entropy over evaluated positions (mean, to keep same scale as OpenVLA path)
        probs = F.softmax(action_logits, dim=-1)
        step_entropy = -(probs * step_logprobs).sum(dim=-1)  # [B, max_A]
        # Average over actual length (avoid divide by zero)
        length_denom = length_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        entropy = (step_entropy * length_mask).sum(dim=1, keepdim=True) / length_denom  # [B, 1]

        # Value head at final prompt position (position prompt_len - 1)
        last_hidden = outputs.hidden_states[-1]  # [B, full_len, H]
        hidden_features = last_hidden[:, prompt_len - 1]  # [B, H]
        values = self.value_head(hidden_features)  # [B, 1]

        return logprobs, entropy, values

    # ------------------------------------------------------------------
    # Value-only forward pass
    # ------------------------------------------------------------------
    def get_value(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Value-only forward pass (no action generation).

        Args:
            input_ids: [B, prompt_len] ending at boa token.
            attention_mask: [B, prompt_len]
        Returns:
            values: [B, 1]
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, L, H]
        hidden_features = last_hidden[:, -1]  # [B, H] — last token (should be boa)
        values = self.value_head(hidden_features)  # [B, 1]
        return values

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Return action dimensionality (always 7 for 7-DOF robot arm)."""
        return self.ACTION_DIM

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> dict:
        """Return action normalization statistics. Set by the policy wrapper."""
        if hasattr(self, '_norm_stats') and self._norm_stats is not None:
            return self._norm_stats
        return None

    def set_action_stats(self, norm_stats: dict) -> None:
        """Set action normalization statistics (q01/q99/mean/std)."""
        self._norm_stats = norm_stats
