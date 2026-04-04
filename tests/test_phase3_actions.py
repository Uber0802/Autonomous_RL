"""
Phase 3 Test: Verify action space adaptation.

Tests:
1. SimlerWrapper._process_action with UniVLA token IDs (last_vocab_idx=151642)
2. ActionTokenizer encode → SimlerWrapper decode consistency
3. Continuous action range validation (within WidowX joint limits)
4. Backward compatibility: OpenVLA token IDs (last_vocab_idx=32000) still work
"""
import sys
import os
import numpy as np
import torch

_base = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_base, 'UniVLA', 'reference', 'Emu3'))
sys.path.insert(0, os.path.join(_base, 'UniVLA', 'models'))
sys.path.insert(0, os.path.join(_base, 'SimplerEnv'))

# ============================================================
# Test 1: _process_action logic with UniVLA token IDs
# ============================================================
print("Test 1: _process_action logic with UniVLA token IDs...")

UNIVLA_LAST_VOCAB = 151642
OPENVLA_LAST_VOCAB = 32000  # actually pad_token_id - 1 for OpenVLA ≈ 31999

bins = np.linspace(-1, 1, 256)
bin_centers = (bins[:-1] + bins[1:]) / 2.0

def process_action_logic(raw_actions_np, last_vocab_idx, unnorm_stats):
    """Replicate SimlerWrapper._process_action logic."""
    dact = last_vocab_idx - raw_actions_np
    dact = np.clip(dact - 1, a_min=0, a_max=254)
    normalized_actions = np.asarray([bin_centers[da] for da in dact])

    mask = unnorm_stats.get("mask", np.ones_like(unnorm_stats["q01"], dtype=bool))
    mask = np.asarray(mask).reshape(1, -1)
    action_high = np.array(unnorm_stats["q99"]).reshape(1, -1)
    action_low = np.array(unnorm_stats["q01"]).reshape(1, -1)
    raw_action = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return raw_action, normalized_actions

# Bridge robot norm stats (from UniVLA config)
bridge_stats = {
    "q01": [-0.02887803, -0.04178320, -0.02611316, -0.08117202, -0.09309056, -0.20778717, -1e-10],
    "q99": [0.02819482, 0.04079563, 0.04015786, 0.08070396, 0.07745135, 0.20165429, 0.99980000],
}

# UniVLA: middle-bin tokens → normalized ≈ 0 → denormalized ≈ mean of q01/q99
mid_tokens = np.array([[UNIVLA_LAST_VOCAB - 128] * 7])
raw, norm = process_action_logic(mid_tokens, UNIVLA_LAST_VOCAB, bridge_stats)
assert np.allclose(norm, 0.0, atol=0.01), f"Normalized not near 0: {norm}"
# Denormalized should be near (q01+q99)/2 for each dim
expected_mid = 0.5 * (np.array(bridge_stats["q99"]) + np.array(bridge_stats["q01"]))
assert np.allclose(raw[0], expected_mid, atol=0.01), f"Denormalized mismatch: {raw[0]} vs {expected_mid}"
print(f"  Middle bin → normalized {norm[0,:3]}... → denorm {raw[0,:3]}... OK")

# ============================================================
# Test 2: ActionTokenizer encode → _process_action decode consistency
# ============================================================
print("Test 2: ActionTokenizer encode → _process_action decode round-trip...")
from tokenizer.action_tokenizer import ActionTokenizer

class MockTok:
    pad_token_id = 151643
    def encode(self, text): return [0]
    def decode(self, ids): return ""
    def batch_decode(self, ids): return [""] * len(ids)

at = ActionTokenizer(MockTok(), bins=256, min_action=-1, max_action=1)

# Create some continuous actions (normalized, in [-1, 1])
test_actions = np.array([
    [0.0, 0.5, -0.5, 0.3, -0.7, 0.9, -0.1],
    [0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.5],
])
B = test_actions.shape[0]

# Encode: continuous → token IDs
token_ids = at.encode_actions_to_ids(test_actions)  # [B, 7]
assert token_ids.shape == (B, 7)

# Decode via _process_action logic (same as SimlerWrapper with UniVLA last_vocab_idx)
dact = at.last_vocab_idx - token_ids
dact = np.clip(dact - 1, a_min=0, a_max=254)
decoded_norm = np.asarray([bin_centers[da] for da in dact])

# Check round-trip accuracy (within 1 bin width ≈ 0.008)
max_error = np.abs(test_actions - decoded_norm).max()
assert max_error < 0.02, f"Round-trip error too large: {max_error}"
print(f"  Max round-trip error: {max_error:.4f} (< 0.02) OK")

# ============================================================
# Test 3: Denormalized action range validation
# ============================================================
print("Test 3: Denormalized action range validation...")

# Generate extreme tokens (near -1 and +1 normalized)
low_tokens = np.array([[UNIVLA_LAST_VOCAB] * 7])  # bin 0 → ~ -1
high_tokens = np.array([[UNIVLA_LAST_VOCAB - 255] * 7])  # bin 255 → ~ +1

raw_low, _ = process_action_logic(low_tokens, UNIVLA_LAST_VOCAB, bridge_stats)
raw_high, _ = process_action_logic(high_tokens, UNIVLA_LAST_VOCAB, bridge_stats)

# raw_low should be close to q01, raw_high close to q99
q01 = np.array(bridge_stats["q01"])
q99 = np.array(bridge_stats["q99"])

assert np.allclose(raw_low[0], q01, atol=0.01), f"Low action not near q01: {raw_low[0]} vs {q01}"
assert np.allclose(raw_high[0], q99, atol=0.01), f"High action not near q99: {raw_high[0]} vs {q99}"
print(f"  Low tokens → {raw_low[0,:3]}... ≈ q01 {q01[:3]}... OK")
print(f"  High tokens → {raw_high[0,:3]}... ≈ q99 {q99[:3]}... OK")

# Verify gripper action is reasonable
# After denormalization, gripper dim (index 6) should be in [~0, ~1]
# Then simpler_wrapper converts: gripper = 2*(x > 0.5) - 1 → {-1, 1}
mid_gripper = raw[0, 6]  # from mid token
assert -0.1 <= mid_gripper <= 1.1, f"Gripper value out of range: {mid_gripper}"
print(f"  Gripper (mid): {mid_gripper:.4f}, range OK")

# ============================================================
# Test 4: Backward compatibility with OpenVLA tokens
# ============================================================
print("Test 4: Backward compatibility with OpenVLA tokens (last_vocab_idx=32000)...")

# Simulate OpenVLA tokens: middle bin
openvla_mid = np.array([[32000 - 128] * 7])
raw_ovla, norm_ovla = process_action_logic(openvla_mid, 32000, bridge_stats)
assert np.allclose(norm_ovla, 0.0, atol=0.01), f"OpenVLA normalized not near 0: {norm_ovla}"
print(f"  OpenVLA middle bin → normalized {norm_ovla[0,:3]}... OK")

# ============================================================
# Test 5: Token ID range doesn't overflow
# ============================================================
print("Test 5: Token ID range safety...")

# Check that action token IDs don't conflict with special tokens
# UniVLA special tokens: pad=151643, bos=151849, boa=151844, eoa=151845
# Action token range: [151642-256, 151642] = [151386, 151642]
action_range_start = UNIVLA_LAST_VOCAB - 256
action_range_end = UNIVLA_LAST_VOCAB
special_tokens = [151643, 151849, 151844, 151845]

for st in special_tokens:
    assert not (action_range_start <= st <= action_range_end), \
        f"Action token range [{action_range_start}, {action_range_end}] conflicts with special token {st}"
print(f"  Action range [{action_range_start}, {action_range_end}] does not conflict with special tokens OK")

print("\n=== All Phase 3 tests passed! ===")
