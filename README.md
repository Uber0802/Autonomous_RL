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

Choose **Option A** (CUDA 12.1, e.g. A100/H100) or **Option B** (CUDA 12.8, e.g. Blackwell/B-series/newer Ada).

### Option A: CUDA 12.1 (A100 / H100 / older Ada)

```bash
# --- Core ---
pip install "setuptools<70.0.0"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# --- Flash Attention (prebuilt wheel for CUDA 12 + torch 2.2 + python 3.10) ---
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Option B: CUDA 12.8 (Blackwell / newer GPUs)

```bash
pip install "setuptools<70.0.0"
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation --no-cache-dir
```

### Common packages (both options)

```bash
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

# --- TensorFlow (required by prismatic/droid_utils) ---
pip install tensorflow==2.15.0 tensorflow-graphics==2021.12.3

# --- Vision / Misc ---
pip install timm==0.9.10 einops pillow
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"
pip install datasets==3.3.2
pip install huggingface_hub jsonlines json-numpy protobuf draccus==0.8.0
pip install rich matplotlib

# --- Install local packages (editable) ---
cd openvla && pip install -e . && cd ..
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..

# --- Verify transformers version was not changed by pip ---
pip install transformers==4.40.1
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

## 5. Download ManiSkill Assets

ManiSkill assets come from two sources:
- **Bundled assets** (carrot models etc.) — included in the ManiSkill package under `ManiSkill/mani_skill/assets/`
- **Downloaded assets** (bridge scenes, robot models) — stored in `~/.maniskill/data/`

Download the required assets:

```bash
python -m mani_skill.utils.download_asset bridge_v2_real2sim
```

### Understanding the asset path error

If you see:
```
RuntimeError: filesystem error: cannot make canonical path: No such file or directory
[.../assets/carrot/more_carrot/001_carrot_simpler/textured.glb]
```

This is because `pick_place_multi.py` line 23 uses a **relative path**:
```python
CARROT_DATASET_DIR = Path(__file__).parent / ".." / ".." / ".." / ".." / "assets" / "carrot"
```

This resolves to `ManiSkill/mani_skill/assets/carrot/` relative to the installed package location.
The carrot asset files ship with ManiSkill (they have `textured.dae`, not `.glb`).

**The segfault is NOT about missing assets — it's about Vulkan** (see Troubleshooting below).
The fallback loading order in the code is: `.obj` → `.dae` → `.glb`.
It reaches `.glb` only when Vulkan fails before the file check runs properly.

## 6. Vulkan Setup (Required for GPU Rendering)

ManiSkill uses SAPIEN which requires Vulkan for GPU-accelerated rendering. This requires:
1. **Vulkan loader** (`libvulkan.so.1`) — the runtime library
2. **NVIDIA Vulkan ICD** (`libnvidia-vulkan-producer.so`) — tells Vulkan to use your GPU
3. **ICD JSON file** (`nvidia_icd.json`) — points Vulkan loader to the ICD library

### Step 1: Install Vulkan + NVIDIA Vulkan producer

```bash
# Install Vulkan loader
sudo apt-get update && sudo apt-get install -y libvulkan1

# Install NVIDIA Vulkan producer (CRITICAL — bundled inside libnvidia-gl)
# Replace 570 with your driver version (check: nvidia-smi | head -3)
sudo apt-get install -y libnvidia-gl-570
```

If `apt-get install libnvidia-gl-XXX` fails with `Invalid cross-device link` (common in Docker):
```bash
# Extract manually without dpkg install
cd /tmp
apt-get download libnvidia-gl-570
mkdir -p extract && dpkg-deb -x libnvidia-gl-570*.deb extract/
sudo cp extract/usr/lib/x86_64-linux-gnu/libnvidia-vulkan-producer.so* /lib/x86_64-linux-gnu/
sudo ldconfig
```

Verify the producer library is available:
```bash
ldconfig -p | grep vulkan-producer
# Should output: libnvidia-vulkan-producer.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libnvidia-vulkan-producer.so
```

### Step 2: Set up ICD JSON

```bash
# Check if ICD file already exists
ls /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null
```

If it exists, export it:
```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

If no ICD file exists, create one:
```bash
sudo mkdir -p /usr/share/vulkan/icd.d
sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json > /dev/null <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
EOF
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### Step 3: Verify Vulkan works

```bash
python -c "
import sapien
engine = sapien.Engine()
print('SAPIEN Vulkan OK')
"
```

Add the `VK_ICD_FILENAMES` export to `~/.bashrc` or your training scripts.

## 7. Verify Installation

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

## 8. Run Training

```bash
# (Optional) Warm-up test — only to verify env/model/RL pipeline works (~2 min, no wandb)
# You can skip this and go straight to full training
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

**`CUDA error: no kernel image is available for execution on the device`:**
- Your GPU requires a newer CUDA toolkit (e.g. Blackwell/B-series, Ada with CUDA 12.8)
- Replace torch + flash-attn:
  ```bash
  pip uninstall -y torch torchvision torchaudio flash-attn
  pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  pip install flash-attn --no-build-isolation --no-cache-dir
  pip install transformers==4.40.1  # restore — pip may have changed it
  ```

**`PrismaticProcessor.from_pretrained()` fails:**
- Ensure `transformers==4.40.1` (not 4.44+)
- Ensure `auto_map` is in both `config.json` and `preprocessor_config.json`
- Ensure `.py` files are in the checkpoint directory

**`No module named 'prismatic'`:**
- Set `PYTHONPATH=$PWD/openvla:$PYTHONPATH` before running

**`No module named 'tensorflow_graphics'`:**
- `pip install tensorflow==2.15.0 tensorflow-graphics==2021.12.3`

**Vulkan errors / `Segmentation fault (core dumped)` when creating env:**
- See Section 6 above for full Vulkan setup
- The `textured.glb` path error is a **symptom** of Vulkan failure, not the root cause
- Most common cause: `libnvidia-vulkan-producer.so` is missing
  - Fix: `sudo apt-get install -y libnvidia-gl-570` (match your driver version)
  - Verify: `ldconfig -p | grep vulkan-producer`
- Set: `export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`

**`dpkg: unable to make backup link ... Invalid cross-device link` during apt install:**
- Docker/container filesystem limitation. Extract the library manually:
  ```bash
  cd /tmp && apt-get download libnvidia-gl-570
  mkdir -p extract && dpkg-deb -x libnvidia-gl-570*.deb extract/
  sudo cp extract/usr/lib/x86_64-linux-gnu/libnvidia-vulkan-producer.so* /lib/x86_64-linux-gnu/
  sudo ldconfig
  ```

**`bridgev2 not found` when downloading assets:**
- The correct asset name is `bridge_v2_real2sim` (with underscores), not `bridgev2`
- Run: `python -m mani_skill.utils.download_asset bridge_v2_real2sim`

**CUDA OOM:**
- Reduce `--num_envs` (default 64) or `--buffer_inferbatch` (default 32)
- The model uses ~15 GB VRAM with LoRA rank 32

**`OSError: Directory not empty` at exit:**
- Non-critical cleanup error for memmap buffer dir. Can be ignored or manually deleted.
