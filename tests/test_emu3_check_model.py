"""
Self-contained sanity check for the Emu3 UniVLA path.

Run this on ANY machine (local or remote) to verify the model + FAST
tokenizer + image processor + decode pipeline produces reasonable
actions on a fixed Bridge scene image.

Usage:
    cd UniVLA_RL
    CUDA_VISIBLE_DEVICES=0 python tests/test_emu3_check_model.py [path_to_image.png]

If no image path is given, uses the first frame of an existing rollout
video. If that's not available either, falls back to a deterministic
Gaussian noise image with seed 42.

Compares against reference numbers from the local machine (where the
warm-up was confirmed working). Differences > 5% on z indicate a
tokenizer / vocabulary mismatch.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMU3_REF = os.path.join(REPO_ROOT, "UniVLA", "reference", "Emu3")
sys.path.insert(0, EMU3_REF)

CKPT = os.path.join(REPO_ROOT, "checkpoints/univla-emu3-raw/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K")
VISION_VQ = os.path.join(REPO_ROOT, "checkpoints/emu3-vision-tokenizer")
FAST_TOK = os.path.join(REPO_ROOT, "checkpoints/fast-bridge-t5-s50")

# Reference numbers from running this same script on the local machine
# with the correct (Bridge-fit, vocab=1024) FAST tokenizer + 256² image.
REFERENCE = {
    "fast_vocab_size": 1024,
    "fast_scale": 50.0,
    "fast_min_token": -112,
    # 4 known scenes from earlier rollout (env0..env3.mp4 first frames)
    # z values within these tolerances → model works as expected
    "z_mean_range": (-0.015, +0.005),       # should reach down on average
    "z_std_min":     0.001,                  # should vary across scenes
    "x_abs_max":     0.030,                  # within q01/q99
}


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default=None, help="Optional input image (PNG/JPG)")
    p.add_argument("--cuda", default="0")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)

    section("[1] Check FAST tokenizer config")
    cfg_path = os.path.join(FAST_TOK, "processor_config.json")
    cfg = json.load(open(cfg_path))
    print(json.dumps(cfg, indent=2))

    fast_vocab = cfg.get("vocab_size")
    fast_scale = cfg.get("scale")
    fast_mintok = cfg.get("min_token")
    print()
    ok = True
    for name, got, want in [
        ("vocab_size", fast_vocab, REFERENCE["fast_vocab_size"]),
        ("scale", fast_scale, REFERENCE["fast_scale"]),
        ("min_token", fast_mintok, REFERENCE["fast_min_token"]),
    ]:
        if got != want:
            print(f"  ✗ {name}: got {got!r}, expected {want!r}")
            ok = False
        else:
            print(f"  ✓ {name}: {got}")
    if not ok:
        print("\n  >>> FAST tokenizer mismatch detected. <<<")
        print("  This is almost certainly the cause of garbage actions.")
        print("  Even if the rest of the script runs, the decoded actions will be meaningless.")
        print("  Fix: replace checkpoints/fast-bridge-t5-s50/ with the Bridge-fit variant")
        print("       (vocab_size=1024, scale=50, min_token=-112).")
        print()

    section("[2] Loading model + tokenizers")
    from transformers import AutoModel, AutoImageProcessor, AutoProcessor, GenerationConfig
    from transformers import LogitsProcessor
    from transformers.generation import LogitsProcessorList
    from emu3.mllm.modeling_emu3 import Emu3MoE
    from emu3.mllm.processing_emu3 import Emu3Processor

    sys.path.insert(0, CKPT)
    from tokenization_emu3 import Emu3Tokenizer

    model = Emu3MoE.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    ).to(device).eval()
    print(f"  model: hidden={model.config.hidden_size}, vocab={model.config.vocab_size}")

    # CRITICAL: must match the policy. Use left-padding for batched generation.
    tokenizer = Emu3Tokenizer.from_pretrained(
        CKPT, model_max_length=model.config.max_position_embeddings,
        padding_side="left", use_fast=False,
    )
    image_processor = AutoImageProcessor.from_pretrained(VISION_VQ, trust_remote_code=True)
    target = 256 * 256
    image_processor.min_pixels = target
    image_processor.max_pixels = target
    image_processor.size = {"min_pixels": target, "max_pixels": target}

    image_tokenizer = AutoModel.from_pretrained(VISION_VQ, trust_remote_code=True).to(device).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    fast_tokenizer = AutoProcessor.from_pretrained(FAST_TOK, trust_remote_code=True)

    last_token_id = tokenizer.pad_token_id - 1  # 151642
    eoa_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eoa_token)  # 151845
    allowed = list(range(last_token_id - fast_tokenizer.vocab_size + 1, last_token_id + 1)) + [eoa_token_id]
    print(f"  last_token_id={last_token_id}, eoa={eoa_token_id}, allowed range="
          f"[{allowed[0]}, {allowed[-2]}] + eoa")

    class ActionIDCon(LogitsProcessor):
        def __init__(self, ids):
            self.ids = ids
        def __call__(self, inp, scores):
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[:, self.ids] = True
            scores[~mask] = -float("inf")
            return scores

    gen_cfg = GenerationConfig(
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=eoa_token_id,
        do_sample=False,
    )

    section("[3] Load test images")
    candidate_paths = []
    if args.image:
        candidate_paths.append(args.image)
    candidate_paths += [
        "/tmp/new_env0.mp4_0.png",
        "/tmp/new_env1.mp4_0.png",
        "/tmp/new_env2.mp4_0.png",
        "/tmp/new_env3.mp4_0.png",
        "/tmp/bridge_scene.png",
    ]
    images = []
    labels = []
    for p in candidate_paths:
        if os.path.exists(p):
            images.append(Image.open(p).convert("RGB"))
            labels.append(os.path.basename(p))
    if not images:
        print("  No real Bridge frames found, falling back to deterministic noise")
        rng = np.random.default_rng(42)
        for i in range(4):
            arr = (rng.standard_normal((480, 640, 3)) * 30 + 128).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr))
            labels.append(f"noise_{i}")
    print(f"  loaded {len(images)} images: {labels}")

    section("[4] Run model on each image and decode")
    z_values = []
    x_values = []
    grip_values = []
    for img, label in zip(images, labels):
        pv = image_processor(img, return_tensors="pt")["pixel_values"].to(device, image_tokenizer.dtype)
        vc = image_tokenizer.encode(pv)
        if vc.ndim == 3:
            vc = vc.unsqueeze(0)

        inp = processor.video_process(
            text="put ketchup bottle on yellow_plate",
            video_tokens=vc,
            context_frames=1, frames=1,
            return_tensors="pt", mode="VLA", padding="longest",
        )
        out = model.generate(
            inp.input_ids.to(device), gen_cfg,
            max_new_tokens=50,
            logits_processor=LogitsProcessorList([ActionIDCon(allowed)]),
            attention_mask=inp.attention_mask.to(device),
        )
        gen = out[:, inp.input_ids.shape[-1]:]
        if gen[0, -1].item() == eoa_token_id:
            gen = gen[:, :-1]
        n_gen = gen.shape[1]
        bpe_ids = (last_token_id - gen[0]).cpu().tolist()

        # Manual decode (matches what UniVLAPolicy._decode_actions does now)
        from scipy.fft import idct
        try:
            decoded_str = fast_tokenizer.bpe_tokenizer.decode(bpe_ids)
            coeffs = np.array(list(map(ord, decoded_str)), dtype=np.float64) + fast_tokenizer.min_token
            max_len = 10 * 7
            if len(coeffs) > max_len:
                coeffs = coeffs[:max_len]
            elif len(coeffs) < max_len:
                coeffs = np.pad(coeffs, (0, max_len - len(coeffs)), mode="constant")
            coeffs = coeffs.reshape(10, 7)
            chunk = idct(coeffs / fast_tokenizer.scale, axis=0, norm="ortho")
            first_norm = chunk[0]
        except Exception as e:
            print(f"  {label}: DECODE ERROR: {e}")
            first_norm = np.zeros(7)

        # Bridge_robot q01/q99 (matches UniVLAPolicy hardcoded fallback)
        q01 = np.array([-0.0289, -0.0418, -0.0261, -0.0812, -0.0931, -0.2078, -1e-10])
        q99 = np.array([+0.0282, +0.0408, +0.0402, +0.0807, +0.0775, +0.2017, +0.9998])
        first_phys = 0.5 * (first_norm + 1) * (q99 - q01) + q01

        z_values.append(first_phys[2])
        x_values.append(first_phys[0])
        grip_values.append(first_phys[6])
        print(f"  {label}: gen_len={n_gen:2}  "
              f"x={first_phys[0]:+.5f} y={first_phys[1]:+.5f} z={first_phys[2]:+.5f}  "
              f"g={first_phys[6]:+.4f}")

    section("[5] Verdict")
    z_mean = float(np.mean(z_values))
    z_std = float(np.std(z_values))
    x_abs_max = float(np.max(np.abs(x_values)))
    print(f"  z mean: {z_mean:+.5f}  (reference: -0.005..-0.001 — should reach down)")
    print(f"  z std:  {z_std:.5f}   (reference: > 0.001 — must be scene-dependent)")
    print(f"  |x| max: {x_abs_max:.5f}  (reference: < 0.030 — should be in q01/q99 range)")

    z_low, z_high = REFERENCE["z_mean_range"]
    checks = {
        "z mean reaches down": z_low <= z_mean <= z_high,
        "z is scene-dependent": z_std >= REFERENCE["z_std_min"],
        "x within physical range": x_abs_max <= REFERENCE["x_abs_max"],
    }
    print()
    all_ok = True
    for name, passed in checks.items():
        mark = "✓" if passed else "✗"
        print(f"  {mark} {name}")
        if not passed:
            all_ok = False

    print()
    if all_ok and ok:
        print("  >>> Model + tokenizer + decoder pipeline looks healthy. <<<")
        print("  If RL training still produces bad rollouts, suspect either:")
        print("    - smaller-than-training image grid (try --vla_image_pixels=262144)")
        print("    - prompt format mismatch (check task_description text)")
    else:
        print("  >>> One or more checks failed. <<<")
        if not ok:
            print("  Most likely cause: FAST tokenizer mismatch (Layer 1 above).")
        else:
            print("  Compare your z/x numbers against the reference and look for systematic")
            print("  shifts. Send the output of this script for further diagnosis.")


if __name__ == "__main__":
    main()
