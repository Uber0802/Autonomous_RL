"""
Smoke test: load Emu3 UniVLA + VisionVQ + FAST tokenizer, run inference on
a real Bridge scene image, decode the action, and check if it's sensible.

Key questions to answer:
1. Does the model load without errors?
2. Does VQ encoding work on real RGB images?
3. Does the model generate action tokens (constrained to FAST vocab)?
4. Does FAST decode produce a sensible action (z-down, gripper-open)?

If yes → proceed with infrastructure work.
If no → diagnose and fall back to OpenVLA.
"""
import os, sys, json, numpy as np, torch, cv2
from PIL import Image

# Add Emu3 reference to path
EMU3_REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "UniVLA", "reference", "Emu3"))
sys.path.insert(0, EMU3_REF)

CKPT = "checkpoints/univla-emu3-raw/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K"
VQ_CKPT = "checkpoints/emu3-vision-tokenizer"
FAST_CKPT = "checkpoints/fast-bridge-t5-s50"  # Bridge-specific FAST: vocab=1024, scale=50, min_token=-112

device = torch.device("cuda:1")
print("=" * 70)
print("Loading components...")
print("=" * 70)

from transformers import AutoModel, AutoImageProcessor, AutoProcessor, GenerationConfig
from transformers.generation import LogitsProcessorList
from emu3.mllm.modeling_emu3 import Emu3MoE
from emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from emu3.mllm.processing_emu3 import Emu3Processor
from transformers import LogitsProcessor


class ActionIDConstraintLogitsProcessor(LogitsProcessor):
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


print(f"\n[1/4] Loading Emu3MoE from {CKPT}")
model = Emu3MoE.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
model.to(device).eval()
print(f"  Model loaded: vocab_size={model.config.vocab_size}, hidden={model.config.hidden_size}")

print(f"\n[2/4] Loading Emu3 tokenizer from {CKPT}")
tokenizer = Emu3Tokenizer.from_pretrained(
    CKPT,
    model_max_length=model.config.max_position_embeddings,
    padding_side="right",
    use_fast=False,
)
print(f"  pad_token_id: {tokenizer.pad_token_id}")
print(f"  bos_token_id: {tokenizer.bos_token_id}")
print(f"  eos_token_id: {tokenizer.eos_token_id}")

print(f"\n[3/4] Loading Emu3-VisionTokenizer (VQ encoder) from {VQ_CKPT}")
image_processor = AutoImageProcessor.from_pretrained(VQ_CKPT, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VQ_CKPT, trust_remote_code=True).to(device, dtype=torch.bfloat16).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
print(f"  VQ encoder loaded")

print(f"\n[4/4] Loading FAST action tokenizer from {FAST_CKPT}")
action_tokenizer = AutoProcessor.from_pretrained(FAST_CKPT, trust_remote_code=True)
print(f"  FAST vocab_size: {action_tokenizer.vocab_size}")
print(f"  FAST scale: {action_tokenizer.scale}")
print(f"  FAST min_token: {action_tokenizer.min_token}")

# Action token range
EOA_TOKEN_ID = 151845
last_token_id = tokenizer.pad_token_id - 1  # 151642
allowed_token_ids = list(range(last_token_id - action_tokenizer.vocab_size, last_token_id + 1)) + [EOA_TOKEN_ID]
print(f"\n  Action token range: [{last_token_id - action_tokenizer.vocab_size}, {last_token_id}]")
print(f"  EOA token: {EOA_TOKEN_ID}")
action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)


# ===== Run inference on real Bridge scenes =====
print("\n" + "=" * 70)
print("Running inference on real Bridge scenes")
print("=" * 70)

# Load 4 real Bridge scene images
imgs_pil = []
prompts = []
for env, task in [
    ("env0", "put carrot on yellow_plate"),
    ("env1", "put carrot on cloth"),
    ("env2", "put kitchen shovel on yellow_plate"),
    ("env3", "put kitchen shovel on cloth"),
]:
    p = f'/tmp/transplant_{env}.mp4_start.png'
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs_pil.append(Image.fromarray(img))
    prompts.append(task)

print(f"\nLoaded {len(imgs_pil)} images, sizes: {[img.size for img in imgs_pil]}")

# Encode images via VQ
print("\nEncoding images via VisionVQ...")
with torch.no_grad():
    pixel_values = image_processor(images=imgs_pil, return_tensors="pt")["pixel_values"].to(device, dtype=torch.bfloat16)
    print(f"  pixel_values: {pixel_values.shape}")
    # Encode to discrete tokens
    video_tokens = image_tokenizer.encode(pixel_values)  # [B, H, W] discrete tokens
    print(f"  video_tokens: {video_tokens.shape}, dtype={video_tokens.dtype}")
    print(f"  token range: [{video_tokens.min().item()}, {video_tokens.max().item()}]")
    # Add time dim: video_tokens shape becomes [B, T=1, H, W]
    video_tokens = video_tokens.unsqueeze(1)
    print(f"  with time dim: {video_tokens.shape}")

# Build VLA prompts
print("\nBuilding VLA prompts...")
kwargs = dict(mode='VLA', padding="longest", context_frames=1)
pos_inputs = processor.video_process(
    text=prompts, video_tokens=video_tokens, gripper_tokens=None,
    frames=1, return_tensors="pt", **kwargs
)
print(f"  input_ids: {pos_inputs.input_ids.shape}")
print(f"  attention_mask: {pos_inputs.attention_mask.shape}")

# Generate
print("\nGenerating action tokens...")
gen_config = GenerationConfig(
    pad_token_id=model.config.pad_token_id,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=EOA_TOKEN_ID,
    do_sample=False,
)
with torch.no_grad():
    output = model.generate(
        pos_inputs.input_ids.to(device),
        gen_config,
        max_new_tokens=50,
        logits_processor=[action_id_processor],
        attention_mask=pos_inputs.attention_mask.to(device),
    )
gen = output[:, pos_inputs.input_ids.shape[-1]:]
print(f"  Generated shape: {gen.shape}")
for i in range(len(prompts)):
    g = gen[i].cpu().tolist()
    if EOA_TOKEN_ID in g:
        g = g[:g.index(EOA_TOKEN_ID)]
    print(f"  env{i}: {len(g)} tokens, first 10: {g[:10]}")

# Decode each environment's action
print("\nDecoding actions via FAST tokenizer...")
last_token_id_tensor = torch.tensor(last_token_id, dtype=gen.dtype, device=gen.device)

for i in range(len(prompts)):
    g_full = gen[i].cpu().tolist()
    # Trim at EOA
    if EOA_TOKEN_ID in g_full:
        g_trim = g_full[:g_full.index(EOA_TOKEN_ID)]
    else:
        g_trim = g_full
    if len(g_trim) == 0:
        print(f"\nenv{i} ({prompts[i]}): EMPTY GENERATION!")
        continue

    # Convert: token_id → FAST BPE id (last_token_id - token_id)
    bpe_ids = [last_token_id - t for t in g_trim]
    try:
        decoded = action_tokenizer.decode([bpe_ids], time_horizon=10, action_dim=7)
        action_chunk = decoded[0]  # [10, 7]
        first_action = action_chunk[0]
        print(f"\nenv{i} ({prompts[i]}):")
        print(f"  First action: {[round(float(v),5) for v in first_action]}")
        print(f"    xyz: [{first_action[0]:+.4f}, {first_action[1]:+.4f}, {first_action[2]:+.4f}] m")
        print(f"    rot: [{first_action[3]:+.4f}, {first_action[4]:+.4f}, {first_action[5]:+.4f}] rad")
        print(f"    gripper: {first_action[6]:+.4f}")

        # Sanity check
        z = first_action[2]
        grip = first_action[6]
        z_ok = z < 0  # going down toward objects
        grip_ok = grip > 0  # open (or at least > 0)
        print(f"    z direction: {'DOWN (good)' if z_ok else 'UP (bad - flying away)'}")
        print(f"    gripper:     {'OPEN (good)' if grip_ok else 'CLOSED (bad - cant grasp)'}")
    except Exception as e:
        print(f"\nenv{i}: DECODE FAILED: {e}")

print("\n" + "=" * 70)
print("Smoke test complete.")
print("=" * 70)
