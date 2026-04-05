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

The install follows the same order as AutoRL: `torch → openvla → tyro → datasets → flash-attn → ManiSkill → SimplerEnv`. CUDA version is auto-detected.

### Option A: One-step setup (recommended)

```bash
bash setup.sh
```

### Option B: Manual install

Choose CUDA 12.1 (A100/H100) or CUDA 12.8 (Blackwell/newer Ada):

```bash
# --- Step 1: PyTorch ---
# CUDA 12.1:
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.8:
# pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# --- Step 2: openvla (pulls transformers==4.40.1, peft, tensorflow, etc.) ---
cd openvla && pip install -e . && cd ..

# --- Step 3: tyro + datasets ---
pip install -U tyro
pip install datasets==3.3.2

# --- Step 4: Flash Attention ---
# CUDA 12.1:
wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && rm flash_attn-*.whl
# CUDA 12.8:
# pip install psutil && pip install flash-attn --no-build-isolation --no-cache-dir

# --- Step 5: ManiSkill & SimplerEnv ---
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

## 6. Run Training

```bash
bash train_univla_full.sh
```

Or warm-up test first (optional, verifies pipeline works):
```bash
bash train_univla.sh
```

## Troubleshooting

### Vulkan / Segfault Issues

ManiSkill uses SAPIEN which requires Vulkan for GPU-accelerated rendering. Most machines with NVIDIA drivers have this pre-configured (AutoRL works without explicit Vulkan setup). If you get segfaults or Vulkan errors, the system is missing:
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

### Other Errors

**`undefined symbol: _ZNK3c105Error4whatEv` in `flash_attn_2_cuda`:**
- flash-attn was compiled against a different torch version than what's installed
- Fix: rebuild flash-attn against your current torch:
  ```bash
  pip uninstall -y flash-attn
  pip install psutil && pip install flash-attn --no-build-isolation --no-cache-dir
  ```

**`CUDA error: no kernel image is available for execution on the device`:**
- GPU requires newer CUDA toolkit. Re-run `setup.sh` (auto-detects CUDA version), or manually:
  ```bash
  pip uninstall -y torch torchvision torchaudio flash-attn
  pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  pip install psutil && pip install flash-attn --no-build-isolation --no-cache-dir
  ```

**`PrismaticProcessor.from_pretrained()` fails:**
- Ensure `transformers==4.40.1` (not 4.44+)
- Ensure `python patch_checkpoint.py` was run

**`No module named 'prismatic'`:**
- Training scripts set PYTHONPATH automatically. If running manually: `PYTHONPATH=$PWD/openvla:$PYTHONPATH`

**`No module named 'tensorflow_graphics'`:**
- Should be installed by `openvla`'s pyproject.toml. If missing: `pip install tensorflow-graphics==2021.12.3`

**`bridgev2 not found` when downloading assets:**
- Correct name: `python -m mani_skill.utils.download_asset bridge_v2_real2sim`

**CUDA OOM:**
- Reduce `--num_envs` (default 64) or `--buffer_inferbatch` (default 32)
