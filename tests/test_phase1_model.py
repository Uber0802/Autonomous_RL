"""
Phase 1 Test: Verify Emu3MoEForRL model loads and forward passes produce correct shapes.

This test uses a SMALL config (not a pretrained checkpoint) to verify the architecture.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'UniVLA', 'reference', 'Emu3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'UniVLA'))

import torch
import torch.nn as nn

# Test 1: Import the model class
print("Test 1: Import Emu3MoEForRL...")
from models.modeling_emu3_rl import Emu3MoEForRL, ValueHead
print("  OK")

# Test 2: Verify ValueHead shapes
print("Test 2: ValueHead shapes...")
hidden_size = 256  # small for testing
vh = ValueHead(hidden_size)
x = torch.randn(2, hidden_size)
out = vh(x)
assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
print(f"  ValueHead({hidden_size}) -> {out.shape} OK")

# Test 3: Create a small Emu3MoEForRL model from config
print("Test 3: Create small Emu3MoEForRL from config...")
from emu3.mllm.configuration_emu3 import Emu3Config

# Minimal config for testing (not a real model, just verifying architecture)
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
# Disable action experts for this basic test (we're testing RL wrapper, not flow matching)
small_config.action_experts = False

model = Emu3MoEForRL(small_config, vh_mode="a0")
model.eval()

# Setup action tokens (simulate: 256 bins, last_vocab_idx = pad_token_id - 1 = 1022)
last_vocab_idx = small_config.pad_token_id - 1  # 1022
n_bins = 256
model.setup_action_tokens(last_vocab_idx=last_vocab_idx, n_action_bins=n_bins, eoa_token_id=1)
print(f"  Model created. Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Action token range: [{last_vocab_idx - n_bins}, {last_vocab_idx}]")

# Test 4: get_value forward pass
print("Test 4: get_value forward pass...")
B = 2
seq_len = 10
input_ids = torch.randint(0, 500, (B, seq_len))
attention_mask = torch.ones(B, seq_len, dtype=torch.long)

with torch.no_grad():
    values = model.get_value(input_ids, attention_mask)
assert values.shape == (B, 1), f"Expected ({B}, 1), got {values.shape}"
print(f"  get_value -> values {values.shape} OK")

# Test 5: evaluate_action forward pass
print("Test 5: evaluate_action forward pass...")
action_len = 7
# Create action token IDs in valid range
action_tokens = torch.randint(last_vocab_idx - n_bins, last_vocab_idx, (B, action_len))
eoa = torch.full((B, 1), 1, dtype=torch.long)  # eoa_token_id = 1

# Full sequence: prompt + action_tokens + eoa
full_input = torch.cat([input_ids, action_tokens, eoa], dim=1)
full_mask = torch.ones(B, full_input.shape[1], dtype=torch.long)

with torch.no_grad():
    logprobs, entropy, values = model.evaluate_action(
        full_input, full_mask, action_tokens, action_len
    )
assert logprobs.shape == (B, 1), f"logprobs: expected ({B}, 1), got {logprobs.shape}"
assert entropy.shape == (B, 1), f"entropy: expected ({B}, 1), got {entropy.shape}"
assert values.shape == (B, 1), f"values: expected ({B}, 1), got {values.shape}"
print(f"  evaluate_action -> logprobs {logprobs.shape}, entropy {entropy.shape}, values {values.shape} OK")

# Test 6: predict_action_batch
print("Test 6: predict_action_batch...")
with torch.no_grad():
    values, gen_ids, logprobs = model.predict_action_batch(
        input_ids, attention_mask,
        max_action_tokens=action_len,
        do_sample=False,
    )
assert values.shape == (B, 1), f"values: expected ({B}, 1), got {values.shape}"
assert gen_ids.shape[0] == B, f"gen_ids batch: expected {B}, got {gen_ids.shape[0]}"
assert gen_ids.shape[1] <= action_len, f"gen_ids len: expected <= {action_len}, got {gen_ids.shape[1]}"
assert logprobs.shape == (B, 1), f"logprobs: expected ({B}, 1), got {logprobs.shape}"
print(f"  predict_action_batch -> values {values.shape}, gen_ids {gen_ids.shape}, logprobs {logprobs.shape} OK")

# Test 7: Gradient flow through value head
print("Test 7: Gradient flow through value head...")
model.train()
values = model.get_value(input_ids, attention_mask)
loss = values.sum()
loss.backward()

vh_grad = model.value_head.head_l1.weight.grad
assert vh_grad is not None, "No gradient on value head!"
assert vh_grad.abs().sum() > 0, "Zero gradient on value head!"
print(f"  Value head gradient norm: {vh_grad.norm():.6f} OK")

print("\n=== All Phase 1 tests passed! ===")
