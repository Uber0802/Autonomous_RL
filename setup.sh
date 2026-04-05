#!/bin/bash
set -e  # Exit on error
set -x  # Print commands as they run

# ===== Detect CUDA version =====
# Auto-detect from nvidia-smi, default to cu121
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "12.1")
CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VER | cut -d. -f2)
echo "Detected CUDA $CUDA_VER"

# ===== Base Dependencies =====
pip install "setuptools<70.0.0"

if [ "$CUDA_MAJOR" -ge 13 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]); then
    echo "===== Installing for CUDA >= 12.8 ====="
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install flash-attn --no-build-isolation --no-cache-dir
else
    echo "===== Installing for CUDA < 12.8 (cu121) ====="
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
    wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi

# ===== Core Dependencies =====
# transformers 4.40.1 required by PrismaticProcessor.from_pretrained()
# Note: 4.44+ breaks PrismaticProcessor auto-loading
pip install transformers==4.40.1
pip install peft==0.11.1
pip install accelerate>=1.0.0
pip install sentencepiece==0.1.99
pip install tiktoken==0.6.0
pip install pillow

# ===== AutoRL / RL4VLA Dependencies =====
pip install "numpy<2.0.0" gymnasium==0.29.1 tyro wandb tqdm transforms3d sapien==3.0.0.b1
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"
pip install einops timm==0.9.10 rich matplotlib
pip install datasets==3.3.2
pip install huggingface_hub jsonlines json-numpy protobuf draccus==0.8.0

# ===== TensorFlow (required by prismatic/droid_utils) =====
pip install tensorflow==2.15.0 tensorflow-graphics==2021.12.3

# ===== Install local packages (editable) =====
cd openvla && pip install -e . && cd ..
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..

# ===== Restore transformers version (pip may have changed it) =====
pip install transformers==4.40.1

# ===== Download Checkpoint =====
echo "===== Downloading qwbu/univla-7b checkpoint (~14 GB) ====="
huggingface-cli download qwbu/univla-7b --local-dir checkpoints/univla-7b

# ===== Patch Checkpoint =====
echo "===== Patching checkpoint for Prismatic auto-loading ====="
python patch_checkpoint.py

# ===== Download ManiSkill Assets =====
echo "===== Downloading ManiSkill bridge_v2_real2sim assets ====="
python -m mani_skill.utils.download_asset bridge_v2_real2sim

# ===== Vulkan Setup =====
echo "===== Checking Vulkan setup ====="
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    echo "Vulkan ICD found at /etc/vulkan/icd.d/nvidia_icd.json"
    echo "Add to your shell: export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json"
elif [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    echo "Vulkan ICD found at /usr/share/vulkan/icd.d/nvidia_icd.json"
    echo "Add to your shell: export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json"
else
    echo "WARNING: No Vulkan ICD found. GPU rendering will fail."
    echo "Install with: sudo apt-get install -y libnvidia-gl-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)"
    echo "Or see README.md Section 6 for manual ICD setup."
fi

# ===== Verify Installation =====
echo "===== Verifying installation ====="
PYTHONPATH=$PWD/openvla:$PYTHONPATH python -c "
import torch
print(f'torch: {torch.__version__}, CUDA: {torch.version.cuda}')
import transformers; print(f'transformers: {transformers.__version__}')
assert transformers.__version__ == '4.40.1', f'ERROR: transformers must be 4.40.1, got {transformers.__version__}'
import flash_attn; print(f'flash_attn: {flash_attn.__version__}')
import peft; print(f'peft: {peft.__version__}')
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
print('prismatic: OK')
import mani_skill; print('mani_skill: OK')
print('All imports passed!')
"

echo "===== Setup complete ====="
echo ""
echo "Next steps:"
echo "  1. export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json"
echo "  2. bash train_univla.sh        # warm-up test (optional)"
echo "  3. bash train_univla_full.sh   # full training"
