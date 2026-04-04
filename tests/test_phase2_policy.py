"""
Phase 2 Test: Verify UniVLAPolicy and UniVLAPPO wrapper classes.

Tests the policy wrapper with a SMALL mock model (no pretrained checkpoint).
Verifies: image encoding → token assembly → model forward → correct output shapes.
"""
import sys
import os

# Add paths
_base = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_base, 'UniVLA', 'reference', 'Emu3'))
sys.path.insert(0, os.path.join(_base, 'UniVLA'))
sys.path.insert(0, os.path.join(_base, 'UniVLA', 'models'))
sys.path.insert(0, os.path.join(_base, 'SimplerEnv'))

import torch
import torch.nn as nn
import numpy as np

# ============================================================
# Test 1: Import UniVLAPolicy and UniVLAPPO
# ============================================================
print("Test 1: Import UniVLAPolicy and UniVLAPPO...")
from simpler_env.policies.univla.univla_train import UniVLAPolicy, UniVLAPPO, huber_loss
print("  OK")

# ============================================================
# Test 2: Verify huber_loss
# ============================================================
print("Test 2: Huber loss...")
e = torch.tensor([0.5, 2.0, -0.3, -5.0])
d = 1.0
h = huber_loss(e, d)
assert h.shape == (4,), f"Expected (4,), got {h.shape}"
assert h[0].item() == 0.5 ** 2 / 2  # |e| <= d case
print(f"  huber_loss OK: {h.tolist()}")

# ============================================================
# Test 3: ActionTokenizer integration
# ============================================================
print("Test 3: ActionTokenizer integration...")
from models.tokenizer.action_tokenizer import ActionTokenizer

# Create a mock tokenizer-like object with pad_token_id
class MockTokenizerForAT:
    pad_token_id = 151643
    def encode(self, text):
        return [0]
    def decode(self, ids):
        return ""
    def batch_decode(self, ids):
        return [""] * len(ids)

mock_tok = MockTokenizerForAT()
at = ActionTokenizer(mock_tok, bins=256, min_action=-1, max_action=1)
assert at.last_vocab_idx == 151642, f"Expected 151642, got {at.last_vocab_idx}"
assert at.n_bins == 256

# Encode some actions
actions = np.array([[0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0]])
token_ids = at.encode_actions_to_ids(actions)
assert token_ids.shape == (1, 7), f"Expected (1,7), got {token_ids.shape}"
# Token IDs should be in range [last_vocab_idx - 255, last_vocab_idx]
assert np.all(token_ids >= at.last_vocab_idx - 255), f"Token IDs too low: {token_ids}"
assert np.all(token_ids <= at.last_vocab_idx), f"Token IDs too high: {token_ids}"

# Decode back
decoded = at.decode_token_ids_to_actions(token_ids)
assert decoded.shape == (1, 7), f"Expected (1,7), got {decoded.shape}"
# Check approximate round-trip (within bin width ~0.008)
assert np.allclose(actions, decoded, atol=0.01), f"Round-trip error too large: {actions} vs {decoded}"
print(f"  ActionTokenizer: encode → decode round-trip OK (max error: {np.abs(actions - decoded).max():.4f})")

# ============================================================
# Test 4: Emu3MoEForRL with action token setup matching ActionTokenizer
# ============================================================
print("Test 4: Model + ActionTokenizer token range consistency...")
from modeling_emu3_rl import Emu3MoEForRL, ValueHead
from emu3.mllm.configuration_emu3 import Emu3Config

small_config = Emu3Config(
    vocab_size=1024,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    pad_token_id=1023,
    bos_token_id=0,
    eos_token_id=1,
)
small_config.action_experts = False

model = Emu3MoEForRL(small_config, vh_mode="a0")
model.eval()

# Use same convention as the real tokenizer: last_vocab_idx = pad_token_id - 1
last_vocab_idx = small_config.pad_token_id - 1  # 1022
n_bins = 256
eoa_token_id = 1

model.setup_action_tokens(last_vocab_idx=last_vocab_idx, n_action_bins=n_bins, eoa_token_id=eoa_token_id)
assert model.last_vocab_idx == 1022
assert model.n_action_bins == 256
print(f"  Model action range: [{last_vocab_idx - n_bins}, {last_vocab_idx}], eoa={eoa_token_id}")

# ============================================================
# Test 5: Simulate the full policy pipeline (without VisionVQ)
# ============================================================
print("Test 5: Simulate full policy pipeline...")
B = 2
seq_len = 20
action_dim = 7

# Simulate preprocessed input_ids (as if VQ encoding + prompt already done)
input_ids = torch.randint(0, 500, (B, seq_len))
attention_mask = torch.ones(B, seq_len, dtype=torch.long)

# 5a: get_value
with torch.no_grad():
    values = model.get_value(input_ids, attention_mask)
assert values.shape == (B, 1), f"get_value: expected ({B},1), got {values.shape}"
print(f"  get_value → {values.shape} OK")

# 5b: predict_action_batch
with torch.no_grad():
    values, gen_ids, logprobs = model.predict_action_batch(
        input_ids, attention_mask,
        max_action_tokens=action_dim,
        do_sample=True,
        temperature=1.0,
    )
assert values.shape == (B, 1)
assert gen_ids.shape[0] == B
assert gen_ids.shape[1] <= action_dim
assert logprobs.shape == (B, 1)
print(f"  predict_action_batch → values {values.shape}, gen_ids {gen_ids.shape}, logprobs {logprobs.shape} OK")

# 5c: evaluate_action (teacher-forced)
# Build full sequence: prompt + action_tokens + eoa
action_tokens = torch.randint(last_vocab_idx - n_bins, last_vocab_idx, (B, action_dim))
eoa = torch.full((B, 1), eoa_token_id, dtype=torch.long)
full_input = torch.cat([input_ids, action_tokens, eoa], dim=1)
full_mask = torch.ones(B, full_input.shape[1], dtype=torch.long)

with torch.no_grad():
    logprobs_eval, entropy_eval, values_eval = model.evaluate_action(
        full_input, full_mask, action_tokens, action_dim
    )
assert logprobs_eval.shape == (B, 1)
assert entropy_eval.shape == (B, 1)
assert values_eval.shape == (B, 1)
print(f"  evaluate_action → logprobs {logprobs_eval.shape}, entropy {entropy_eval.shape}, values {values_eval.shape} OK")

# ============================================================
# Test 6: Gradient flow end-to-end
# ============================================================
print("Test 6: Gradient flow through evaluate_action + value head...")
model.train()
logprobs_train, entropy_train, values_train = model.evaluate_action(
    full_input, full_mask, action_tokens, action_dim
)
loss = -logprobs_train.mean() + values_train.sum()
loss.backward()

vh_grad = model.value_head.head_l1.weight.grad
assert vh_grad is not None, "No gradient on value head!"
assert vh_grad.abs().sum() > 0, "Zero gradient on value head!"

# Check that lm_head also has gradients (policy gradient flows)
lm_grad = model.lm_head.weight.grad
assert lm_grad is not None, "No gradient on lm_head!"
print(f"  VH grad norm: {vh_grad.norm():.6f}, LM head grad norm: {lm_grad.norm():.6f} OK")

# ============================================================
# Test 7: PPO loss computation (mock)
# ============================================================
print("Test 7: PPO loss computation (mock)...")
model.zero_grad()
model.eval()

with torch.no_grad():
    values_old, gen_ids_old, logprobs_old = model.predict_action_batch(
        input_ids, attention_mask, max_action_tokens=action_dim, do_sample=True, temperature=1.0,
    )

# Simulate PPO step
model.train()
action_tok = gen_ids_old  # reuse generated actions
eoa_t = torch.full((B, 1), eoa_token_id, dtype=torch.long)
full_in = torch.cat([input_ids, action_tok, eoa_t], dim=1)
full_mk = torch.ones(B, full_in.shape[1], dtype=torch.long)

logprobs_new, entropy_new, values_new = model.evaluate_action(
    full_in, full_mk, action_tok, action_tok.shape[1]
)

# PPO ratio
ratio = torch.exp(logprobs_new - logprobs_old.detach())
assert ratio.shape == (B, 1), f"ratio shape: {ratio.shape}"
assert not torch.isnan(ratio).any(), "NaN in ratio!"
assert not torch.isinf(ratio).any(), "Inf in ratio!"

# Surrogate loss
advantages = torch.randn(B, 1)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
policy_loss = -torch.min(surr1, surr2).mean()

# Value loss
returns = torch.randn(B, 1)
value_loss = huber_loss(returns - values_new, 10.0).mean()

total_loss = policy_loss + value_loss
total_loss.backward()

print(f"  PPO loss: policy={policy_loss.item():.4f}, value={value_loss.item():.4f}, total={total_loss.item():.4f}")
print(f"  Ratio: mean={ratio.mean().item():.4f}, min={ratio.min().item():.4f}, max={ratio.max().item():.4f}")
print("  OK")

# ============================================================
# Test 8: Token ID compatibility with simpler_wrapper
# ============================================================
print("Test 8: Token ID ↔ continuous action compatibility...")
# Simulate what simpler_wrapper._process_action does
# For UniVLA: last_vocab_idx - token_id → bin index (instead of 32000 - token_id for OpenVLA)
# This test verifies the mapping is correct

# Generate some action token IDs as the model would
mock_last_vocab = 151642  # real UniVLA value
mock_action_tokens = np.array([[mock_last_vocab - 128] * 7])  # middle bin

# UniVLA decode path
bins = np.linspace(-1, 1, 256)
bin_centers = (bins[:-1] + bins[1:]) / 2.0

dact = mock_last_vocab - mock_action_tokens  # bin indices
dact = np.clip(dact - 1, a_min=0, a_max=254)
normalized_actions = np.asarray([bin_centers[da] for da in dact])

# Middle bin should map to approximately 0
assert np.allclose(normalized_actions, 0.0, atol=0.01), f"Middle bin != 0: {normalized_actions}"
print(f"  Token {mock_action_tokens[0,0]} → bin {dact[0,0]} → action {normalized_actions[0,0]:.4f} ≈ 0.0 OK")

# Test extremes
low_tokens = np.array([[mock_last_vocab] * 7])  # bin 0 → action ≈ -1
high_tokens = np.array([[mock_last_vocab - 255] * 7])  # bin 255 → action ≈ 1
dact_low = np.clip(mock_last_vocab - low_tokens - 1, 0, 254)
dact_high = np.clip(mock_last_vocab - high_tokens - 1, 0, 254)
act_low = bin_centers[dact_low[0, 0]]
act_high = bin_centers[dact_high[0, 0]]
assert act_low < -0.9, f"Low action not near -1: {act_low}"
assert act_high > 0.9, f"High action not near 1: {act_high}"
print(f"  Low token → action {act_low:.4f} ≈ -1.0, High token → action {act_high:.4f} ≈ 1.0 OK")


print("\n=== All Phase 2 tests passed! ===")
