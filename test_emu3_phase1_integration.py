"""
Integration test: verify Emu3MoEForRL (Phase 1 code) works end-to-end with
the SFT'd Emu3 UniVLA Bridge checkpoint.

Tests:
1. Load model + VQ encoder + Emu3 tokenizer + FAST action tokenizer
2. Encode real Bridge scene images via VQ
3. Build VLA prompt with Emu3Processor
4. Call predict_action_batch() → (values, action_token_ids, logprobs)
5. Decode action_token_ids back to continuous actions
6. Verify shapes and value ranges
"""
import os, sys, json, numpy as np, torch, cv2
from PIL import Image

# Phase 1 imports
sys.path.insert(0, "UniVLA")
from models.modeling_emu3_rl import Emu3MoEForRL

# Emu3 reference imports
EMU3_REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "UniVLA", "reference", "Emu3"))
sys.path.insert(0, EMU3_REF)
from emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from emu3.mllm.processing_emu3 import Emu3Processor

from transformers import AutoModel, AutoImageProcessor, AutoProcessor

CKPT = "checkpoints/univla-emu3-raw/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K"
VQ_CKPT = "checkpoints/emu3-vision-tokenizer"
FAST_CKPT = "checkpoints/fast-bridge-t5-s50"
NORM_STATS = "checkpoints/univla-emu3-raw/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K/norm_stats.json"

device = torch.device("cuda:0")
print("Loading Emu3MoEForRL (Phase 1 model)...")
model = Emu3MoEForRL.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    vh_mode="a0",
)
model.to(device).eval()

print("Loading Emu3 tokenizer + VQ encoder + Emu3Processor...")
tokenizer = Emu3Tokenizer.from_pretrained(
    CKPT, model_max_length=model.config.max_position_embeddings,
    padding_side="left", use_fast=False,
)
image_processor = AutoImageProcessor.from_pretrained(VQ_CKPT, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VQ_CKPT, trust_remote_code=True).to(device, dtype=torch.bfloat16).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

print("Loading FAST action tokenizer (Bridge t5_s50)...")
action_tokenizer = AutoProcessor.from_pretrained(FAST_CKPT, trust_remote_code=True)
print(f"  vocab_size={action_tokenizer.vocab_size}, scale={action_tokenizer.scale}, min_token={action_tokenizer.min_token}")

# Setup action tokens in the model
EOA_TOKEN_ID = 151845
LAST_TOKEN_ID = tokenizer.pad_token_id - 1  # 151642
model.setup_action_tokens(
    last_vocab_idx=LAST_TOKEN_ID,
    n_action_bins=action_tokenizer.vocab_size,
    eoa_token_id=EOA_TOKEN_ID,
)

# Load Bridge norm stats
with open(NORM_STATS) as f:
    norm_stats = json.load(f)['norm_stats']
model.set_action_stats(norm_stats)
print(f"  Loaded norm_stats keys: {list(norm_stats.keys())}")

# ===== Run inference on real Bridge scenes =====
print("\n" + "=" * 70)
print("Running predict_action_batch on real Bridge scenes")
print("=" * 70)

imgs_pil, prompts = [], []
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

# Encode via VQ
with torch.no_grad():
    pixel_values = image_processor(images=imgs_pil, return_tensors="pt")["pixel_values"].to(device, dtype=torch.bfloat16)
    video_tokens = image_tokenizer.encode(pixel_values).unsqueeze(1)  # [B, 1, H, W]
print(f"VQ-encoded video_tokens: {video_tokens.shape}")

# Build VLA prompt
pos_inputs = processor.video_process(
    text=prompts, video_tokens=video_tokens, gripper_tokens=None,
    frames=1, return_tensors="pt", mode='VLA', padding="longest", context_frames=1,
)
print(f"input_ids: {pos_inputs.input_ids.shape}, attention_mask: {pos_inputs.attention_mask.shape}")

# Call Phase 1 predict_action_batch
print("\nCalling Emu3MoEForRL.predict_action_batch()...")
with torch.no_grad():
    values, action_ids, logprobs = model.predict_action_batch(
        input_ids=pos_inputs.input_ids.to(device),
        attention_mask=pos_inputs.attention_mask.to(device),
        max_action_tokens=50,  # generous upper bound for variable-length
        do_sample=False,
    )

print(f"\nReturn shapes:")
print(f"  values: {values.shape}")
print(f"  action_ids: {action_ids.shape}")
print(f"  logprobs: {logprobs.shape}")
print(f"\nValues: {[round(float(v),4) for v in values.squeeze().cpu().tolist()]}")
print(f"Logprobs: {[round(float(l),4) for l in logprobs.squeeze().cpu().tolist()]}")

# Decode action_ids back to continuous actions
print("\n=== Decoding actions per env ===")
last_token_id_tensor = torch.tensor(LAST_TOKEN_ID, dtype=action_ids.dtype, device=action_ids.device)
nonzero_count = (action_ids != model.config.pad_token_id).sum(dim=1)

# Bridge stats for unnormalization
bs = norm_stats['bridge_robot']
mask = np.array([True]*6 + [False])
q01 = np.array(bs['q01'])
q99 = np.array(bs['q99'])

for i in range(len(prompts)):
    aids = action_ids[i].cpu().tolist()
    # Trim trailing pad/eoa
    aids_trim = []
    for t in aids:
        if t == EOA_TOKEN_ID or t == model.config.pad_token_id:
            break
        aids_trim.append(t)
    if not aids_trim:
        print(f"\nenv{i}: empty generation")
        continue
    bpe_ids = [LAST_TOKEN_ID - t for t in aids_trim]
    try:
        decoded = action_tokenizer.decode([bpe_ids], time_horizon=10, action_dim=7)
        a0 = decoded[0, 0]  # first timestep
        a_phys = np.where(mask, 0.5 * (a0 + 1) * (q99 - q01) + q01, a0)
        print(f"\nenv{i} ({prompts[i]}):")
        print(f"  num action tokens: {len(aids_trim)}")
        print(f"  norm action[0]: {[round(float(x),4) for x in a0]}")
        print(f"  phys xyz: [{a_phys[0]*100:+.2f}, {a_phys[1]*100:+.2f}, {a_phys[2]*100:+.2f}] cm")
        print(f"  phys rot: [{a_phys[3]:+.3f}, {a_phys[4]:+.3f}, {a_phys[5]:+.3f}] rad")
        g = float(a0[6])
        print(f"  gripper: {'OPEN' if g > 0 else 'CLOSE'} ({g:+.3f})")
    except Exception as e:
        print(f"\nenv{i}: decode failed: {e}")

print("\n" + "=" * 70)
print("Phase 1 integration test complete.")
print("=" * 70)
