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
# UniVLA-7b SFT checkpoint (~14 GB) — fine-tuned on SimplerEnv/Bridge tasks
# The base univla-7b acts randomly on Bridge tasks; use the SFT version for RL warm-start.
huggingface-cli download qwbu/univla-7b-224-sft-simpler-bridge --local-dir checkpoints/univla-7b-sft-bridge
```

## 4. Patch Checkpoint for Prismatic Loading

`qwbu/univla-7b` is missing `auto_map` fields that `PrismaticProcessor.from_pretrained()` requires. Run this once after downloading:

```bash
python patch_checkpoint.py --ckpt_dir checkpoints/univla-7b-sft-bridge
```

Or do it manually:

```bash
# Copy prismatic source files into checkpoint directory
cp openvla/prismatic/extern/hf/configuration_prismatic.py checkpoints/univla-7b-sft-bridge/
cp openvla/prismatic/extern/hf/modeling_prismatic.py checkpoints/univla-7b-sft-bridge/
cp openvla/prismatic/extern/hf/processing_prismatic.py checkpoints/univla-7b-sft-bridge/
```

Then add `auto_map` to `checkpoints/univla-7b-sft-bridge/config.json`:
```json
"auto_map": {
    "AutoConfig": "configuration_prismatic.OpenVLAConfig",
    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"
}
```

And add `auto_map` to `checkpoints/univla-7b-sft-bridge/preprocessor_config.json`:
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

There are two paths depending on which UniVLA variant you want to use.

### Path A — Prismatic UniVLA (drop-in OpenVLA replacement)

```bash
# Full PPO training with the SFT'd Prismatic UniVLA
bash train_univla_full.sh

# Or warm-up smoke test (1 episode, 4 envs)
bash train_univla.sh
```

Uses `qwbu/univla-7b-224-sft-simpler-bridge`. Same architecture, tokenizer and
action space as OpenVLA — only `--vla_path` and `--vla_unnorm_key` differ.

### Path B — Emu3 UniVLA (FAST tokenization)

```bash
# Production training (102 GB GPU, num_envs=64, 384² image)
bash train_univla_emu3_full.sh

# Or warm-up smoke test (1 episode, 2 envs, 256² image)
bash train_univla_emu3.sh
```

Uses `Yuqi1997/UniVLA/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K` with the Emu3
backbone, VisionVQ image encoding, and the FAST BPE action tokenizer. Three
extra artifacts must be in `checkpoints/`:

| Path | Source | Notes |
|---|---|---|
| `checkpoints/univla-emu3-raw/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K/` | `Yuqi1997/UniVLA` | ~14 GB, downloaded by `setup.sh` Step 7b |
| `checkpoints/emu3-vision-tokenizer/` | `BAAI/Emu3-VisionTokenizer` | ~300 MB, downloaded by `setup.sh` Step 7b |
| `checkpoints/fast-bridge-t5-s50/` | **manual transfer** (see below) | 52 KB, **not on HuggingFace** |

**FAST tokenizer warning**: `setup.sh` currently downloads
`physical-intelligence/fast` (vocab=2048, scale=10) into the right directory
name, but this is the **wrong variant**. The model was trained with the
Bridge-fit `fast_bridge_t5_s50` (`vocab=1024, scale=50, min_token=-112`),
which has a completely different BPE merge table. Using the wrong variant
silently produces garbage actions.

After running `setup.sh`, **always verify**:

```bash
cat checkpoints/fast-bridge-t5-s50/processor_config.json
# expect: vocab_size: 1024, scale: 50, min_token: -112
```

If you see `vocab_size: 2048`, transfer the correct variant:

```bash
# On the source machine that has the right files (~10 KB)
tar czf /tmp/fast-bridge-t5-s50.tar.gz \
    -C UniVLA_RL/checkpoints fast-bridge-t5-s50

# scp / rsync /tmp/fast-bridge-t5-s50.tar.gz to the target machine, then
cd UniVLA_RL/checkpoints
rm -rf fast-bridge-t5-s50
tar xzof /tmp/fast-bridge-t5-s50.tar.gz   # 'o' avoids ownership errors
```

For full diagnosis run:

```bash
python tests/test_emu3_check_model.py
```

It checks the FAST config, runs inference on real Bridge frames, and prints
a verdict comparing your numbers to the reference. All three checks must
pass before launching training.

### OOM remediation for `train_univla_emu3_full.sh`

If the production script OOMs, drop the knobs **in this order** until it
fits, observing `[mem]` peaks (re-enable with `GPU_MEM_DEBUG=1`). Each
step is roughly equal in cost; stop as soon as it survives one full
episode (rollout → train → eval → render → save).

| Step | Edit | Effect on memory | Effect on quality / speed |
|---|---|---|---|
| 1 | `--vla_image_pixels=147456 → 65536` (384² → 256²) | rollout & train −15 GB each | weaker zero-shot detail; ~1.5× faster |
| 2 | `--alg_gradient_accum=8 → 16` | unchanged | 2× more PPO microsteps per optimizer step (slower, no quality loss) |
| 3 | `--num_envs=64 → 32` | rollout buffer halves (~10 GB) | half as many parallel rollouts, halves PPO sample throughput |
| 4 | `--buffer_inferbatch=2 → 1` | rollout peak −5 GB | sequential generation, halves rollout throughput |
| 5 | `--vla_image_pixels=65536 → 36864` (256² → 192²) | rollout & train −10 GB each | 540 vision tokens (very low); zero-shot may degrade |
| 6 | re-enable grad checkpointing if you removed it (default is on) | train peak −20 GB | 30% slower backward |
| 7 | `--vla_lora_rank=32 → 8` | LoRA params + Adam state ÷4 | weaker fine-tuning capacity |

Conversely, if you have lots of headroom and want to go faster / better:

| Bump (in order) | New value | Memory cost | Why |
|---|---|---|---|
| `--vla_image_pixels` | `262144` (512²) | +20 GB rollout, +25 GB train | training-time grid, attention pattern aligned with pretraining |
| `--buffer_inferbatch` | `4` then `8` | +10 GB then +20 GB rollout | parallel rollout generation |
| `--buffer_minibatch` | `2` then `4` | +10 GB then +20 GB train | bigger PPO minibatch → less variance |
| `--num_envs` | `128` | +10 GB buffer storage | more parallel envs → faster wall-clock |

**Rule of thumb**: keep ≥20 GB headroom on top of the steady-state peak you
see in `[mem]`. The largest single tensor allocation during PPO backward
can be 30-45 GB on its own (the lm_head logits gradient + attention
recompute spike). If `[mem]` shows you using 75 GB, do NOT bump anything.

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
- Ensure `python patch_checkpoint.py --ckpt_dir checkpoints/univla-7b-sft-bridge` was run

**`No module named 'prismatic'`:**
- Training scripts set PYTHONPATH automatically. If running manually: `PYTHONPATH=$PWD/openvla:$PYTHONPATH`

**`No module named 'tensorflow_graphics'`:**
- Should be installed by `openvla`'s pyproject.toml. If missing: `pip install tensorflow-graphics==2021.12.3`

**`bridgev2 not found` when downloading assets:**
- Correct name: `python -m mani_skill.utils.download_asset bridge_v2_real2sim`

**CUDA OOM:**
- Reduce `--num_envs` (default 64) or `--buffer_inferbatch` (default 32)
