"""
Patch qwbu/univla-7b checkpoint for Prismatic auto-loading.

qwbu/univla-7b is missing `auto_map` fields that PrismaticProcessor.from_pretrained()
requires. This script adds them and copies the necessary .py files.

Run once after downloading the checkpoint:
    python patch_checkpoint.py [--ckpt_dir checkpoints/univla-7b]
"""
import json
import shutil
import argparse
from pathlib import Path


def patch(ckpt_dir: str = "checkpoints/univla-7b"):
    ckpt = Path(ckpt_dir)
    assert ckpt.exists(), f"Checkpoint directory not found: {ckpt}"
    prismatic_dir = Path("openvla/prismatic/extern/hf")
    assert prismatic_dir.exists(), f"Prismatic source not found: {prismatic_dir}"

    # 1. Copy .py files
    for py_file in ["configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"]:
        src = prismatic_dir / py_file
        dst = ckpt / py_file
        if not dst.exists():
            shutil.copy(src, dst)
            print(f"  Copied {py_file}")
        else:
            print(f"  {py_file} already exists, skipping")

    # 2. Patch config.json
    config_path = ckpt / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    if "auto_map" not in config:
        config["auto_map"] = {
            "AutoConfig": "configuration_prismatic.OpenVLAConfig",
            "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("  Patched config.json with auto_map")
    else:
        print("  config.json already has auto_map, skipping")

    # 3. Patch preprocessor_config.json
    preproc_path = ckpt / "preprocessor_config.json"
    with open(preproc_path, "r") as f:
        preproc = json.load(f)
    if "auto_map" not in preproc:
        preproc["auto_map"] = {
            "AutoImageProcessor": "processing_prismatic.PrismaticImageProcessor",
        }
        with open(preproc_path, "w") as f:
            json.dump(preproc, f, indent=2)
        print("  Patched preprocessor_config.json with auto_map")
    else:
        print("  preprocessor_config.json already has auto_map, skipping")

    print(f"\nDone! Checkpoint at {ckpt} is ready for use.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="checkpoints/univla-7b")
    args = parser.parse_args()
    patch(args.ckpt_dir)
