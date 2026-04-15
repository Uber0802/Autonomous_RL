import sys, pathlib, json, torch

_cronos = pathlib.Path(__file__).resolve().parents[1]   # cronos_univla/CRONOS
_univla = _cronos.parent / "UniVLA"

sys.path.insert(0, str(_cronos))
sys.path.insert(0, str(_univla))

# --- 1. CRONOS internal imports ---
from envs.wrapper import CronosWrapper
from training.ppo import CronosPPO
from training.buffer import CronosReplayBuffer
print("[OK] CRONOS imports")

# --- 2. UniVLA prismatic ---
from prismatic.extern.hf.modeling_prismatic import (
    UniVLAForActionPredictionWithValueHead,
    VQAllowedLogitsProcessor,
)
print("[OK] UniVLA prismatic (UniVLAForActionPredictionWithValueHead)")

# --- 3. ActionDecoder (TF-free) ---
from univla_action_decoder import ActionDecoder, ActionDecoderHead
ckpt_dir = _univla / "qwbu__univla-7b-224-sft-simpler-bridge"
sd = torch.load(str(ckpt_dir / "action_decoder.pt"), map_location="cpu")
dec = ActionDecoder(window_size=10)
dec.net.load_state_dict(sd)
print(f"[OK] ActionDecoder loaded")

# --- 4. UniVLAPolicy class ---
from simpler_env.policies.univla.univla_train import UniVLAPolicy
print("[OK] UniVLAPolicy importable")

# --- 5. window_size validation ---
proj_key = next(k for k in sd if k.startswith("proj.") and len(sd[k].shape) == 2)
ws = sd[proj_key].shape[0] // 7
assert ws == 10, f"Expected window_size=10, got {ws}"
print(f"[OK] action_decoder.pt window_size={ws} (matches --univla_window_size 10)")

# --- 6. dataset_statistics sanity ---
stats = json.load(open(str(ckpt_dir / "dataset_statistics.json")))
assert "bridge_oxe" in stats
bridge = stats["bridge_oxe"]["action"]
assert len(bridge["q01"]) == 7 and len(bridge["q99"]) == 7 and len(bridge["mask"]) == 7
print("[OK] dataset_statistics.json: bridge_oxe q01/q99/mask valid")

print()
print("=== Pre-flight PASSED — all imports and checkpoint checks succeeded ===")
