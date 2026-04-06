"""
Transplant OpenVLA's 256-bin action head into the UniVLA SFT checkpoint.

Both models share:
- Prismatic backbone (DinoV2 + SigLIP + LLaMA-2 7B)
- LLaMA-2 tokenizer (vocab_size 32064, action bins at 31744-31999)
- Bridge V2 delta-action representation (control_mode identical)

UniVLA's lm_head was tuned for ACT tokens (32001-32032) via the LAM head, leaving
the 256-bin action token outputs near-mean (uninformative). We replace those 256
output projection rows with OpenVLA's well-trained values.

Run once after downloading both checkpoints:
    python transplant_action_head.py \
        --src openvla/openvla-7b \
        --dst checkpoints/univla-7b-sft-bridge
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file


# Action bin tokens occupy indices 31744-31999 (last 256 of effective vocab 32000)
ACTION_TOKEN_START = 31744
ACTION_TOKEN_END = 32000  # exclusive


def find_lm_head_shard(ckpt_dir: str, key: str = "language_model.lm_head.weight"):
    """Locate which safetensors shard contains the lm_head weight."""
    index_path = Path(ckpt_dir) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"].get(key)
    # Single-shard case
    for f in Path(ckpt_dir).glob("*.safetensors"):
        with safe_open(f, framework="pt") as sf:
            if key in sf.keys():
                return f.name
    return None


def load_lm_head_slice(ckpt_dir: str, start: int, end: int) -> torch.Tensor:
    """Load only the [start:end] rows of language_model.lm_head.weight."""
    key = "language_model.lm_head.weight"
    shard = find_lm_head_shard(ckpt_dir, key)
    if shard is None:
        raise RuntimeError(f"Could not find {key} in {ckpt_dir}")
    shard_path = Path(ckpt_dir) / shard
    with safe_open(shard_path, framework="pt") as sf:
        full = sf.get_tensor(key)
    print(f"  Loaded {key} from {shard}: shape={tuple(full.shape)}, dtype={full.dtype}")
    return full[start:end].clone()


def transplant(src_ckpt: str, dst_ckpt: str, dry_run: bool = False):
    """Copy action-bin lm_head rows from src into dst (in-place modification)."""
    print(f"\n=== Source: {src_ckpt} ===")
    if not Path(src_ckpt).exists():
        # HF download
        print(f"  Downloading from HuggingFace...")
        from huggingface_hub import snapshot_download
        src_local = snapshot_download(src_ckpt, allow_patterns=["*.safetensors", "*.json"])
        print(f"  Downloaded to: {src_local}")
    else:
        src_local = src_ckpt

    src_slice = load_lm_head_slice(src_local, ACTION_TOKEN_START, ACTION_TOKEN_END)
    print(f"  Source action-head slice: shape={tuple(src_slice.shape)}, "
          f"mean={src_slice.float().mean().item():+.5f}, std={src_slice.float().std().item():.5f}")

    print(f"\n=== Destination: {dst_ckpt} ===")
    key = "language_model.lm_head.weight"
    shard = find_lm_head_shard(dst_ckpt, key)
    if shard is None:
        raise RuntimeError(f"Could not find {key} in {dst_ckpt}")
    shard_path = Path(dst_ckpt) / shard

    # Load entire shard, modify lm_head, save back
    with safe_open(shard_path, framework="pt") as sf:
        tensors = {k: sf.get_tensor(k) for k in sf.keys()}
        metadata = sf.metadata() or {}

    dst_full = tensors[key]
    print(f"  Loaded shard {shard} ({len(tensors)} tensors)")
    print(f"  {key}: shape={tuple(dst_full.shape)}, dtype={dst_full.dtype}")

    dst_slice_before = dst_full[ACTION_TOKEN_START:ACTION_TOKEN_END].float()
    print(f"  Dst slice BEFORE: mean={dst_slice_before.mean().item():+.5f}, "
          f"std={dst_slice_before.std().item():.5f}")

    # Cast source to dst dtype
    src_slice_cast = src_slice.to(dtype=dst_full.dtype)
    dst_full[ACTION_TOKEN_START:ACTION_TOKEN_END] = src_slice_cast
    tensors[key] = dst_full

    dst_slice_after = dst_full[ACTION_TOKEN_START:ACTION_TOKEN_END].float()
    print(f"  Dst slice AFTER:  mean={dst_slice_after.mean().item():+.5f}, "
          f"std={dst_slice_after.std().item():.5f}")

    diff = (dst_slice_after - src_slice.float()).abs().max().item()
    print(f"  Max diff (after vs source): {diff:.6f}  {'OK' if diff < 1e-3 else 'CAST ERROR'}")

    if dry_run:
        print("\n  [dry run] Skipping write")
        return

    # Backup original shard
    backup = shard_path.with_suffix(shard_path.suffix + ".bak")
    if not backup.exists():
        print(f"  Backing up to {backup.name}")
        shutil.copy2(shard_path, backup)

    print(f"  Writing modified shard to {shard_path}")
    save_file(tensors, str(shard_path), metadata=metadata)
    print("  Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="openvla/openvla-7b",
                        help="Source checkpoint (HF id or local path) — provides the action head")
    parser.add_argument("--dst", default="checkpoints/univla-7b-sft-bridge",
                        help="Destination checkpoint (local path) — receives the action head")
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()
    transplant(args.src, args.dst, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
