"""
Emu3MoE model adapted for RL (PPO) training.

Adds ValueHead and RL-specific methods (predict_action_batch, evaluate_action, get_value)
to the base Emu3MoE model, mirroring OpenVLA's ActionPredictionWithValueHead interface.
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
    """Constrains generation to only produce action tokens or eoa token."""
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
    """3-layer MLP value head for RL, same architecture as OpenVLA's."""
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
    Emu3MoE extended with a value head for RL training (PPO).

    Uses discrete action tokenization (FAST or uniform bins) for compatibility
    with PPO's log-probability requirements.

    Key methods:
        - predict_action_batch: Generate actions and return (values, actions, logprobs)
        - evaluate_action: Teacher-forced evaluation returning (logprobs, entropy, values)
        - get_value: Value-only forward pass
    """

    def __init__(self, config, vh_mode: str = "a0"):
        super().__init__(config)
        self.value_head = ValueHead(config.hidden_size)
        self.vh_mode = vh_mode

        # Action token range: [last_vocab_idx - n_action_bins, last_vocab_idx]
        # These will be set during policy initialization when we know the tokenizer
        self.last_vocab_idx = None
        self.n_action_bins = None
        self.eoa_token_id = 151845  # end of action token

    def setup_action_tokens(self, last_vocab_idx: int, n_action_bins: int, eoa_token_id: int = 151845):
        """Configure action token range. Must be called after loading tokenizer."""
        self.last_vocab_idx = last_vocab_idx
        self.n_action_bins = n_action_bins
        self.eoa_token_id = eoa_token_id
        # Build allowed token IDs for constrained generation
        self.allowed_token_ids = list(range(last_vocab_idx - n_action_bins, last_vocab_idx + 1)) + [eoa_token_id]

    def predict_action_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        max_action_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate action tokens and compute values + logprobs.

        Args:
            input_ids: [B, seq_len] - tokenized input (vision + text prompt)
            attention_mask: [B, seq_len]
            max_action_tokens: number of action tokens to generate
            do_sample: whether to sample (True) or greedy (False)
            temperature: sampling temperature

        Returns:
            values: [B, 1] - value estimates
            generated_ids: [B, max_action_tokens] - generated action token IDs
            logprobs: [B, 1] - sum of log-probabilities over action tokens
        """
        assert self.last_vocab_idx is not None, "Call setup_action_tokens() first"

        batch_size = input_ids.shape[0]
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
            max_new_tokens=max_action_tokens + 1,  # +1 for potential eoa token
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_logits=True,
            logits_processor=LogitsProcessorList([action_logits_processor]),
        )

        # Extract generated action tokens (exclude eoa if present)
        full_generated = output.sequences[:, input_ids.shape[1]:]
        # Remove eoa tokens, keep only action tokens
        # Take first max_action_tokens tokens
        generated_ids = full_generated[:, :max_action_tokens]  # [B, max_action_tokens]

        # Compute logprobs from logits
        # output.logits is a tuple of [B, vocab_size] tensors, one per generated token
        n_generated = min(len(output.logits), max_action_tokens)
        logits_tensor = torch.stack(output.logits[:n_generated], dim=1)  # [B, n_gen, vocab_size]

        # Slice to action token range
        action_start = self.last_vocab_idx - self.n_action_bins
        action_end = self.last_vocab_idx + 1
        action_logits = logits_tensor[:, :, action_start:action_end]  # [B, n_gen, n_bins]
        logprobs_tensor = F.log_softmax(action_logits, dim=-1)  # [B, n_gen, n_bins]

        # Gather logprobs for generated tokens
        idxes = generated_ids[:, :n_generated].unsqueeze(-1) - action_start  # [B, n_gen, 1]
        idxes = idxes.clamp(0, self.n_action_bins)  # safety clamp
        logprobs = torch.gather(logprobs_tensor, 2, idxes).squeeze(-1)  # [B, n_gen]
        logprobs = logprobs.sum(dim=1, keepdim=True)  # [B, 1]

        # Value head: extract hidden states
        # output.hidden_states[i] = tuple of layer hidden states for generation step i
        # output.hidden_states[0][-1] = last layer hidden state at first generation step
        # Shape: [B, seq_len, hidden_size]
        if self.vh_mode == "a0":
            # Use hidden state at the last input position (before first action token)
            first_step_hidden = output.hidden_states[0][-1]  # [B, L, H]
            hidden_features = first_step_hidden[:, -1]  # [B, H]
            values = self.value_head(hidden_features)  # [B, 1]
        else:
            # Default: use last generated token's hidden state
            last_step_hidden = output.hidden_states[-1][-1]  # [B, L, H]
            hidden_features = last_step_hidden[:, -1]  # [B, H]
            values = self.value_head(hidden_features)  # [B, 1]

        return values, generated_ids, logprobs

    def evaluate_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        action_token_ids: torch.LongTensor,
        action_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher-forced evaluation of given action tokens.

        Args:
            input_ids: [B, seq_len] - full sequence including action tokens + eoa
            attention_mask: [B, seq_len]
            action_token_ids: [B, action_len] - the action token IDs to evaluate
            action_len: number of action tokens

        Returns:
            logprobs: [B, 1] - sum of log-probabilities
            entropy: [B, 1] - mean entropy over action positions
            values: [B, 1] - value estimates
        """
        assert self.last_vocab_idx is not None, "Call setup_action_tokens() first"

        # Forward pass with full sequence (prompt + action tokens)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = outputs.hidden_states[-1]  # [B, L, H]

        # Value head: extract from position before first action token
        # Sequence: [... prompt_tokens action_0 ... action_N eoa]
        # Position of last prompt token: -(action_len + 2) for eoa, or -(action_len + 1) if no eoa
        if self.vh_mode == "a0":
            hidden_features = last_hidden_state[:, -(action_len + 2)]  # [B, H]
        else:
            hidden_features = last_hidden_state[:, -2]  # [B, H] (last action token position)
        values = self.value_head(hidden_features)  # [B, 1]

        # Logits at action token positions
        # logits[i] predicts token[i+1], so logits at positions [-(action_len+2): -2] predict action tokens
        action_logits = outputs.logits[:, -(action_len + 2): -2]  # [B, action_len, vocab_size]

        # Slice to action token range
        action_start = self.last_vocab_idx - self.n_action_bins
        action_end = self.last_vocab_idx + 1
        action_logits = action_logits[:, :, action_start:action_end]  # [B, action_len, n_bins]
        logprobs_tensor = F.log_softmax(action_logits, dim=-1)  # [B, action_len, n_bins]

        # Gather logprobs for given action tokens
        idxes = action_token_ids.unsqueeze(-1) - action_start  # [B, action_len, 1]
        idxes = idxes.clamp(0, self.n_action_bins).to(logprobs_tensor.device)
        logprobs = torch.gather(logprobs_tensor, 2, idxes).squeeze(-1)  # [B, action_len]
        logprobs = logprobs.sum(dim=1, keepdim=True)  # [B, 1]

        # Entropy
        probs_tensor = F.softmax(action_logits, dim=-1)  # [B, action_len, n_bins]
        entropy = -(probs_tensor * logprobs_tensor).sum(dim=-1)  # [B, action_len]
        entropy = entropy.mean(dim=-1, keepdim=True)  # [B, 1]

        return logprobs, entropy, values

    def get_value(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Value-only forward pass (no action generation).

        Args:
            input_ids: [B, seq_len] - tokenized input (prompt only, no action tokens)
            attention_mask: [B, seq_len]

        Returns:
            values: [B, 1]
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = outputs.hidden_states[-1]  # [B, L, H]
        hidden_features = last_hidden_state[:, -1]  # [B, H] - last token position
        values = self.value_head(hidden_features)  # [B, 1]

        return values

    def get_action_dim(self, unnorm_key: str = None) -> int:
        """Return action dimensionality (always 7 for robot manipulation)."""
        return 7

    def get_action_stats(self, unnorm_key: str = None) -> dict:
        """
        Return action normalization statistics.
        These are loaded from UniVLA's normalizer configs.
        """
        if hasattr(self, '_norm_stats') and self._norm_stats is not None:
            return self._norm_stats
        return None

    def set_action_stats(self, norm_stats: dict):
        """Set action normalization statistics."""
        self._norm_stats = norm_stats
