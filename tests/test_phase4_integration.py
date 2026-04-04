"""
Phase 4 Test: Verify training loop integration.

Tests:
1. Args dataclass has vla_type and vision_vq_path fields
2. SimlerWrapper accepts last_vocab_idx parameter
3. UniVLAPolicy and UniVLAPPO can be imported from the correct paths
4. The conditional import logic in Runner works for both vla_types
"""
import sys
import os

_base = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_base, 'UniVLA', 'reference', 'Emu3'))
sys.path.insert(0, os.path.join(_base, 'UniVLA', 'models'))
sys.path.insert(0, os.path.join(_base, 'SimplerEnv'))

# ============================================================
# Test 1: Args dataclass has new fields
# ============================================================
print("Test 1: Args dataclass has vla_type and vision_vq_path fields...")
from simpler_env.train_ms3_ppo import Args

args = Args()
assert hasattr(args, 'vla_type'), "Args missing vla_type field"
assert hasattr(args, 'vision_vq_path'), "Args missing vision_vq_path field"
assert args.vla_type == "openvla", f"Default vla_type should be 'openvla', got '{args.vla_type}'"
assert args.vision_vq_path == "", f"Default vision_vq_path should be '', got '{args.vision_vq_path}'"
print(f"  vla_type='{args.vla_type}', vision_vq_path='{args.vision_vq_path}' OK")

# ============================================================
# Test 2: SimlerWrapper accepts last_vocab_idx
# ============================================================
print("Test 2: SimlerWrapper signature accepts last_vocab_idx...")
import inspect
from simpler_env.env.simpler_wrapper import SimlerWrapper

sig = inspect.signature(SimlerWrapper.__init__)
params = list(sig.parameters.keys())
assert 'last_vocab_idx' in params, f"SimlerWrapper missing last_vocab_idx param. Params: {params}"

# Check default value
default = sig.parameters['last_vocab_idx'].default
assert default == 32000, f"Default last_vocab_idx should be 32000, got {default}"
print(f"  SimlerWrapper has last_vocab_idx param (default={default}) OK")

# ============================================================
# Test 3: UniVLA policy imports work from training script path
# ============================================================
print("Test 3: UniVLA policy imports...")
from simpler_env.policies.univla.univla_train import UniVLAPolicy, UniVLAPPO
assert UniVLAPolicy is not None
assert UniVLAPPO is not None
print("  UniVLAPolicy, UniVLAPPO imported OK")

# ============================================================
# Test 4: OpenVLA policy imports still work (if prismatic available)
# ============================================================
print("Test 4: OpenVLA policy imports (backward compatibility)...")
try:
    from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
    assert OpenVLAPolicy is not None
    assert OpenVLAPPO is not None
    print("  OpenVLAPolicy, OpenVLAPPO imported OK")
except ImportError as e:
    print(f"  Skipped (expected in univla_env): {e}")

# ============================================================
# Test 5: Verify the _process_action uses configurable last_vocab_idx
# ============================================================
print("Test 5: SimlerWrapper._process_action uses self.last_vocab_idx...")
import inspect
source = inspect.getsource(SimlerWrapper._process_action)
assert "self.last_vocab_idx" in source, "_process_action doesn't use self.last_vocab_idx!"
assert "32000" not in source, "_process_action still has hardcoded 32000!"
print("  _process_action uses self.last_vocab_idx (no hardcoded 32000) OK")

# ============================================================
# Test 6: Verify Runner conditional import logic is present
# ============================================================
print("Test 6: Runner has conditional vla_type logic...")
# Read source directly since Runner import may fail due to OpenVLA deps
runner_path = os.path.join(_base, 'SimplerEnv', 'simpler_env', 'train_ms3_ppo.py')
with open(runner_path, 'r') as f:
    source = f.read()
assert 'vla_type' in source, "train_ms3_ppo.py doesn't reference vla_type"
assert 'UniVLAPolicy' in source, "train_ms3_ppo.py doesn't reference UniVLAPolicy"
assert 'OpenVLAPolicy' in source, "train_ms3_ppo.py doesn't reference OpenVLAPolicy"
assert 'last_vocab_idx' in source, "train_ms3_ppo.py doesn't pass last_vocab_idx"
print("  Runner has conditional imports for both openvla and univla OK")

print("\n=== All Phase 4 tests passed! ===")
