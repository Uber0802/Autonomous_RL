# UniVLA-RL: RL Training with UniVLA (qwbu/univla-7b)

UniVLA integrated into the AutoRL (RL4VLA) framework for PPO-based fine-tuning on ManiSkill3 / SimplerEnv.

`qwbu/univla-7b` shares the identical Prismatic VLM backbone (DINOSigLIP + LLaMA-2 7B) as `openvla/openvla-7b`, making it a **drop-in replacement** — only the checkpoint path and normalization key change.

## Prerequisites

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU with ≥ 24 GB VRAM (tested on A100/H100)
- CUDA 12.1 driver
- Conda

## 1. Create Conda Environment

```bash
conda create -n univla_env python=3.10 -y
conda activate univla_env
```

## 2. Install Packages

```bash
# --- Core ---
pip install "setuptools<70.0.0"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# --- Transformers stack (must be 4.40.1 — 4.44+ breaks PrismaticProcessor) ---
pip install transformers==4.40.1
pip install peft==0.11.1
pip install accelerate>=1.0.0
pip install sentencepiece==0.1.99
pip install tiktoken==0.6.0

# --- RL / Env ---
pip install "numpy<2.0.0"
pip install gymnasium==0.29.1
pip install sapien==3.0.0.b1
pip install tyro wandb tqdm
pip install transforms3d

# --- Vision / Misc ---
pip install timm==0.9.10 einops pillow
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"
pip install datasets==3.3.2
pip install huggingface_hub jsonlines json-numpy protobuf draccus==0.8.0
pip install rich matplotlib

# --- Flash Attention (prebuilt wheel for CUDA 12 + torch 2.2 + python 3.10) ---
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# --- Install local packages (editable) ---
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..
```

## 3. Download Checkpoint

```bash
# UniVLA-7b pretrained (~14 GB)
huggingface-cli download qwbu/univla-7b --local-dir checkpoints/univla-7b
```

## 4. Patch Checkpoint for Prismatic Loading

`qwbu/univla-7b` is missing `auto_map` fields that `PrismaticProcessor.from_pretrained()` requires. Run this once after downloading:

```bash
python patch_checkpoint.py
```

Or do it manually:

```bash
# Copy prismatic source files into checkpoint directory
cp openvla/prismatic/extern/hf/configuration_prismatic.py checkpoints/univla-7b/
cp openvla/prismatic/extern/hf/modeling_prismatic.py checkpoints/univla-7b/
cp openvla/prismatic/extern/hf/processing_prismatic.py checkpoints/univla-7b/
```

Then add `auto_map` to `checkpoints/univla-7b/config.json`:
```json
"auto_map": {
    "AutoConfig": "configuration_prismatic.OpenVLAConfig",
    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"
}
```

And add `auto_map` to `checkpoints/univla-7b/preprocessor_config.json`:
```json
"auto_map": {
    "AutoImageProcessor": "processing_prismatic.PrismaticImageProcessor"
}
```

## 5. Verify Installation

```bash
PYTHONPATH=$PWD/openvla:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python -c "
import torch
print(f'torch: {torch.__version__}, CUDA: {torch.version.cuda}')
import transformers; print(f'transformers: {transformers.__version__}')
import flash_attn; print(f'flash_attn: {flash_attn.__version__}')
import peft; print(f'peft: {peft.__version__}')
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
print('prismatic: OK')
import mani_skill; print('mani_skill: OK')
print('All imports passed!')
"
```

Expected output:
```
torch: 2.2.0+cu121, CUDA: 12.1
transformers: 4.40.1
flash_attn: 2.7.4.post1
peft: 0.11.1
prismatic: OK
mani_skill: OK
All imports passed!
```

## 6. Run Training

```bash
# Warm-up test (small scale: 4 envs, 20 steps, 1 episode, no wandb)
bash train_univla.sh

# Full training (64 envs, 320 steps, 32 episodes, with wandb)
bash train_univla_full.sh
```

Or run manually:
```bash
cd SimplerEnv
PYTHONPATH=$(dirname $PWD)/openvla:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="univla-experiment" \
  --log="univla-experiment.txt" \
  --wandb_dir=".." \
  --env_id="TwoObjectTwoReceptacle-v1" \
  --vla_path="../checkpoints/univla-7b" \
  --vla_unnorm_key="bridge_oxe" \
  --seed=0
```

## Key Differences from OpenVLA Baseline

| | OpenVLA | UniVLA |
|---|---|---|
| Checkpoint | `openvla/openvla-7b` | `checkpoints/univla-7b` |
| Norm key | `bridge_orig` | `bridge_oxe` |
| Code changes | — | None (same `OpenVLAPolicy`) |
| Extra setup | — | `auto_map` patch + `.py` files in checkpoint dir |
| PYTHONPATH | — | Must include `openvla/` |

## Directory Structure

```
UniVLA_RL/
├── checkpoints/
│   └── univla-7b/           # Downloaded + patched checkpoint
├── ManiSkill/                # ManiSkill3 (editable install)
├── SimplerEnv/               # SimplerEnv + policies + training script
│   └── simpler_env/
│       ├── env/simpler_wrapper.py
│       ├── policies/openvla/openvla_train.py   # Used for both OpenVLA & UniVLA
│       ├── policies/univla/univla_train.py     # For Emu3-based UniVLA (future)
│       ├── train_ms3_ppo.py                    # Main training script
│       └── utils/replay_buffer.py
├── openvla/                  # Prismatic module (loaded via PYTHONPATH)
├── UniVLA/                   # Emu3-based UniVLA code (Phase 1-4, for future use)
├── train_univla.sh           # Warm-up training script
├── train_univla_full.sh      # Full training script
├── patch_checkpoint.py       # Checkpoint patching script
├── setup.sh                  # Package installation script
└── tests/                    # Unit tests (Phase 1-4)
```

## Troubleshooting

**`PrismaticProcessor.from_pretrained()` fails:**
- Ensure `transformers==4.40.1` (not 4.44+)
- Ensure `auto_map` is in both `config.json` and `preprocessor_config.json`
- Ensure `.py` files are in the checkpoint directory

**`No module named 'prismatic'`:**
- Set `PYTHONPATH=$PWD/openvla:$PYTHONPATH` before running

**CUDA OOM:**
- Reduce `--num_envs` (default 64) or `--buffer_inferbatch` (default 32)
- The model uses ~15 GB VRAM with LoRA rank 32

**`OSError: Directory not empty` at exit:**
- Non-critical cleanup error for memmap buffer dir. Can be ignored or manually deleted.
