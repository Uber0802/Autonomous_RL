"""
Smoke test for Emu3-based UniVLA on Bridge task.

Loads the UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K checkpoint, runs inference
on a real Bridge scene image, and verifies that actions are purposeful.

FAST decoder output is in normalized [-1, 1] space (confirmed from
pickle_generation_simplerenv_bridge.py line 165-166). We apply q01/q99
unnormalization to get physical action values.
"""
import os
import sys
import json
import numpy as np
import torch
from PIL import Image

# Paths
CKPT = "checkpoints/univla-emu3-raw/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K"
VISION_VQ = "checkpoints/emu3-vision-tokenizer"
FAST_TOK = "checkpoints/fast-bridge-t5-s50"

# Add Emu3 source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "UniVLA", "reference", "Emu3"))
sys.path.insert(0, CKPT)

from emu3.mllm.modeling_emu3 import Emu3MoE
from emu3.mllm.processing_emu3 import Emu3Processor
from transformers import AutoModel, AutoImageProcessor, AutoProcessor, GenerationConfig
from transformers.generation import LogitsProcessorList
from tokenization_emu3 import Emu3Tokenizer


class ActionIDConstraintLogitsProcessor:
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[:, self.allowed_token_ids] = True
        scores[~mask] = -float("inf")
        return scores


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=== Loading Emu3MoE model ===")
    model = Emu3MoE.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()
    print(f"  Model loaded. vocab={model.config.vocab_size}, hidden={model.config.hidden_size}")

    print("\n=== Loading Emu3 tokenizer ===")
    tokenizer = Emu3Tokenizer.from_pretrained(
        CKPT,
        model_max_length=model.config.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )
    print(f"  vocab_size: {tokenizer.vocab_size}")
    print(f"  boa={tokenizer.boa_token} id={tokenizer.encode(tokenizer.boa_token)[0]}")
    print(f"  eoa={tokenizer.eoa_token} id={tokenizer.encode(tokenizer.eoa_token)[0]}")

    print("\n=== Loading VisionVQ encoder ===")
    image_processor = AutoImageProcessor.from_pretrained(VISION_VQ, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VISION_VQ, trust_remote_code=True).to(device).eval()
    image_processor.min_pixels = 80 * 80

    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    print("\n=== Loading FAST action tokenizer ===")
    action_tokenizer = AutoProcessor.from_pretrained(FAST_TOK, trust_remote_code=True)
    print(f"  vocab_size: {action_tokenizer.vocab_size}")
    print(f"  scale: {action_tokenizer.scale}, min_token: {action_tokenizer.min_token}")

    last_token_id = tokenizer.pad_token_id - 1
    action_token_start = last_token_id - action_tokenizer.vocab_size
    action_token_end = last_token_id
    print(f"  Action token range: [{action_token_start}, {action_token_end}]")

    eoa_token_id = tokenizer.encode(tokenizer.eoa_token)[0]
    allowed_token_ids = list(range(action_token_start, action_token_end + 1)) + [eoa_token_id]
    action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)

    gen_config = GenerationConfig(
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=eoa_token_id,
        do_sample=False,
    )

    # Load normalization stats
    with open(os.path.join(CKPT, "norm_stats.json")) as f:
        ns = json.load(f)["norm_stats"]["bridge_robot"]
    q01 = np.array(ns["q01"])
    q99 = np.array(ns["q99"])
    print(f"\n  q01: {[round(v, 4) for v in q01]}")
    print(f"  q99: {[round(v, 4) for v in q99]}")

    def unnormalize(a_norm):
        return 0.5 * (a_norm + 1) * (q99 - q01) + q01

    print("\n=== Testing on real Bridge scene images ===")
    frames = ["/tmp/new_env0.mp4_0.png", "/tmp/new_env1.mp4_0.png",
              "/tmp/new_env2.mp4_0.png", "/tmp/new_env3.mp4_0.png"]
    prompt_text = "put ketchup bottle on yellow_plate"

    all_actions_norm = []
    all_actions_phys = []
    for i, frame in enumerate(frames):
        if not os.path.exists(frame):
            print(f"  {frame} missing, skipping")
            continue
        img = Image.open(frame).convert("RGB")

        # Encode image to VQ tokens
        pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            video_code = image_tokenizer.encode(pixel_values)  # [1, h, w] discrete tokens
        if video_code.ndim == 3:
            video_code = video_code.unsqueeze(0)  # [1, 1, h, w] treat as 1-frame video
        # video_code shape should be [batch=1, t=1, h, w]
        print(f"  Frame {i}: video_code shape {video_code.shape}")

        pos_inputs = processor.video_process(
            text=prompt_text,
            video_tokens=video_code,
            context_frames=1,
            frames=1,
            return_tensors="pt",
            mode="VLA",
            padding="longest",
        )

        with torch.no_grad():
            outputs = model.generate(
                pos_inputs.input_ids.to(device),
                gen_config,
                max_new_tokens=50,
                logits_processor=LogitsProcessorList([action_id_processor]),
                attention_mask=pos_inputs.attention_mask.to(device),
            )

        # Strip prompt and trailing eoa
        gen = outputs[:, pos_inputs.input_ids.shape[-1]:]
        # Remove eoa if present at end
        if gen[0, -1].item() == eoa_token_id:
            gen = gen[:, :-1]

        # Decode via FAST
        last_token_id_tensor = torch.tensor(last_token_id, dtype=gen.dtype, device=gen.device)
        bpe_ids = last_token_id_tensor - gen
        action_chunk = action_tokenizer.decode(bpe_ids.cpu(), time_horizon=10, action_dim=7)
        # action_chunk: [batch, 10, 7] in normalized [-1, 1] space
        first_norm = np.asarray(action_chunk[0, 0])  # [7]
        first_phys = unnormalize(first_norm)
        all_actions_norm.append(first_norm)
        all_actions_phys.append(first_phys)

        print(f"  Frame {i}:")
        print(f"    normalized: [{', '.join(f'{v:+.3f}' for v in first_norm)}]")
        print(f"    physical:   [{', '.join(f'{v:+.5f}' for v in first_phys)}]")
        print(f"    z_phys={first_phys[2]:+.5f} (want negative, ~[-0.026, 0.04])")
        print(f"    gripper_phys={first_phys[6]:+.4f} (0=closed, 1=open)")

    print("\n=== Analysis (PHYSICAL actions, in Bridge robot frame) ===")
    if len(all_actions_phys) > 0:
        arr = np.stack(all_actions_phys)
        print(f"  z mean:   {arr[:, 2].mean():+.5f}  (want < 0 for reaching down)")
        print(f"  z std:    {arr[:, 2].std():.5f}  (want > 0 for scene dependence)")
        print(f"  z range:  [{arr[:, 2].min():+.5f}, {arr[:, 2].max():+.5f}]  (physical bounds [-0.0261, +0.0402])")
        print(f"  gripper:  {arr[:, 6]}  (want ~1.0 for open)")
        print(f"  xyz std across envs: {arr[:, :3].std(axis=0)}")
        print(f"  full action range per dim:")
        for d in range(7):
            print(f"    dim {d}: [{arr[:, d].min():+.5f}, {arr[:, d].max():+.5f}]  mean={arr[:, d].mean():+.5f}")

        # Compare to AutoRL/OpenVLA baseline
        print("\n  Reference OpenVLA on same scenes:")
        print("    z: ~ -0.01 (down)")
        print("    gripper: ~ 1.0 (open)")
        print("\n  Verdict:")
        z_ok = arr[:, 2].mean() < 0
        grip_ok = arr[:, 6].mean() > 0.5
        scene_ok = arr[:, :3].std(axis=0).sum() > 1e-4
        print(f"    z mean negative:     {'OK' if z_ok else 'FAIL'}")
        print(f"    gripper open:        {'OK' if grip_ok else 'FAIL'}")
        print(f"    scene dependent:     {'OK' if scene_ok else 'FAIL'}")
        if z_ok and grip_ok and scene_ok:
            print("\n  >>> Emu3 UniVLA produces purposeful zero-shot actions. Safe to proceed. <<<")
        else:
            print("\n  >>> Issues detected. Investigate before proceeding. <<<")


if __name__ == "__main__":
    main()
