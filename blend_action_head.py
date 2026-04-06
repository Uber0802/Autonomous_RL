"""
Option C: Blend OpenVLA's action head with UniVLA's at various ratios.
Find the ratio that produces sensible actions on real Bridge scenes.

Tests on real scene images extracted from the rollout videos and reports
which blend produces actions closest to OpenVLA's known-good baseline.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
import numpy as np
import cv2
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

ACTION_TOKEN_START = 31744
ACTION_TOKEN_END = 32000


def find_lm_head_shard(ckpt_dir, key="language_model.lm_head.weight"):
    index_path = Path(ckpt_dir) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"].get(key)
    for f in Path(ckpt_dir).glob("*.safetensors"):
        with safe_open(f, framework="pt") as sf:
            if key in sf.keys():
                return f.name
    return None


def load_lm_head_slice(ckpt_dir, start, end):
    key = "language_model.lm_head.weight"
    shard = find_lm_head_shard(ckpt_dir, key)
    with safe_open(Path(ckpt_dir) / shard, framework="pt") as sf:
        return sf.get_tensor(key)[start:end].clone()


def write_blended(dst_ckpt, blended_slice):
    key = "language_model.lm_head.weight"
    shard = find_lm_head_shard(dst_ckpt, key)
    shard_path = Path(dst_ckpt) / shard
    with safe_open(shard_path, framework="pt") as sf:
        tensors = {k: sf.get_tensor(k) for k in sf.keys()}
        metadata = sf.metadata() or {}
    tensors[key][ACTION_TOKEN_START:ACTION_TOKEN_END] = blended_slice.to(tensors[key].dtype)
    save_file(tensors, str(shard_path), metadata=metadata)


def test_actions(dst_ckpt):
    """Returns avg z direction across 4 real scenes (negative = good, positive = bad)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "openvla"))
    from transformers import AutoTokenizer
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    tok = AutoTokenizer.from_pretrained(dst_ckpt, use_fast=False, padding_side='left')
    ip = PrismaticImageProcessor.from_pretrained(dst_ckpt)
    proc = PrismaticProcessor(ip, tok)
    model = OpenVLAForActionPredictionWithValueHead.from_pretrained(
        dst_ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, device_map='cuda:0', vh_mode='a0',
    )

    imgs = []
    for env in ['env0', 'env1', 'env2', 'env3']:
        img = cv2.imread(f'/tmp/transplant_{env}.mp4_start.png')
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imgs_t = torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2).cuda().to(torch.bfloat16)

    prompts = [
        'In: What action should the robot take to put the carrot on the plate?\nOut: ',
        'In: What action should the robot take to put the carrot on the cloth?\nOut: ',
        'In: What action should the robot take to put the kitchen shovel on the plate?\nOut: ',
        'In: What action should the robot take to put the kitchen shovel on the cloth?\nOut: ',
    ]
    inputs = proc(prompts, imgs_t, padding=True).to('cuda:0', dtype=torch.bfloat16)

    with torch.no_grad():
        _, action_ids, _ = model.predict_action_batch(**inputs, unnorm_key='bridge_oxe', do_sample=False)

    with open(dst_ckpt + '/dataset_statistics.json') as f:
        s = json.load(f)['bridge_oxe']['action']
    bins = np.linspace(-1, 1, 256)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    mask = np.array(s["mask"])
    q99, q01 = np.array(s['q99']), np.array(s['q01'])

    actions = []
    grippers = []
    for i in range(4):
        tids = action_ids[i].cpu().numpy()
        d = np.clip((model.vocab_size - tids) - 1, 0, 254)
        normalized = bin_centers[d]
        a = np.where(mask, 0.5 * (normalized + 1) * (q99 - q01) + q01, normalized)
        actions.append(a)
        grippers.append(a[6])

    actions = np.array(actions)
    return {
        'z_mean': float(actions[:, 2].mean()),
        'z_std': float(actions[:, 2].std()),
        'all_z': actions[:, 2].tolist(),
        'gripper_mean': float(np.mean(grippers)),
        'gripper_open_count': int(sum(g > 0.5 for g in grippers)),
        'all_actions': actions.tolist(),
    }


def main():
    src = "openvla/openvla-7b"
    dst = "checkpoints/univla-7b-sft-bridge"
    backup = Path(dst) / "model-00003-of-00003.safetensors.bak"
    shard = Path(dst) / "model-00003-of-00003.safetensors"
    if not backup.exists():
        print("ERROR: backup not found at", backup)
        return

    print("Loading OpenVLA action head...")
    src_local = snapshot_download(src, allow_patterns=["*.safetensors", "*.json"])
    ov_slice = load_lm_head_slice(src_local, ACTION_TOKEN_START, ACTION_TOKEN_END).float()
    print("Loading UniVLA action head (from backup)...")
    # Use backup as the source of original UniVLA weights
    with safe_open(str(backup), framework="pt") as sf:
        uni_full = sf.get_tensor("language_model.lm_head.weight")
    uni_slice = uni_full[ACTION_TOKEN_START:ACTION_TOKEN_END].float()
    print(f"  OpenVLA: mean={ov_slice.mean():+.5f}, std={ov_slice.std():.5f}")
    print(f"  UniVLA:  mean={uni_slice.mean():+.5f}, std={uni_slice.std():.5f}")

    results = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # alpha = 1.0 → pure OpenVLA, alpha = 0.0 → pure UniVLA
        blended = alpha * ov_slice + (1 - alpha) * uni_slice
        print(f"\n=== alpha = {alpha} (OpenVLA weight) ===")
        print(f"  blended: mean={blended.mean():+.5f}, std={blended.std():.5f}")
        # First restore UniVLA from backup
        shutil.copy2(backup, shard)
        write_blended(dst, blended)
        # Test
        r = test_actions(dst)
        results[alpha] = r
        print(f"  z values across envs:    {[round(z,4) for z in r['all_z']]}")
        print(f"  z mean:                  {r['z_mean']:+.4f} m  ({'GOOD: down' if r['z_mean'] < -0.002 else 'BAD: up' if r['z_mean'] > 0.002 else 'neutral'})")
        print(f"  z stddev (env diversity): {r['z_std']:.4f}")
        print(f"  gripper open count:      {r['gripper_open_count']}/4")

    print("\n\n=== SUMMARY ===")
    print(f"{'alpha':<10}{'z_mean':<12}{'z_std':<12}{'grip_open':<12}")
    for alpha, r in results.items():
        print(f"{alpha:<10}{r['z_mean']:<+12.4f}{r['z_std']:<12.4f}{r['gripper_open_count']:<12}")

    # Restore backup at the end
    shutil.copy2(backup, shard)
    print("\nRestored backup. Use the alpha that gives z_mean < 0 and gripper_open_count = 4.")


if __name__ == "__main__":
    main()
