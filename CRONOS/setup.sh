#!/bin/bash
# setup.sh - Environment setup for CRONOS

set -e

echo "Setting up CRONOS environment..."

# 1. Install core dependencies
pip install "setuptools<70.0.0"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0.0" gymnasium==0.29.1 tyro wandb tqdm transforms3d sapien==3.0.0.b1
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# 2. Install backbone packages in editable mode
pip install -e ../ManiSkill
pip install -e ../SimplerEnv
pip install -e ../openvla

# 3. NVIDIA Vulkan ICD (Fixes IncompatibleDriver)
FOUND_VULKAN=$(find /usr/share/vulkan/icd.d -name "nvidia_icd.json" -print -quit 2>/dev/null)
if [ -n "$FOUND_VULKAN" ]; then
    export VK_ICD_FILENAMES="$FOUND_VULKAN"
    echo "Vulkan ICD found and configured: $FOUND_VULKAN"
else
    echo "WARNING: NVIDIA Vulkan ICD not found in common paths."
fi

# 4. Library Path (Fixes IncompatibleDriver)
NVIDIA_LIB_PATH=$(find /usr/lib -name "libnvidia-glcore.so*" -print -quit 2>/dev/null | xargs dirname)
if [ -n "$NVIDIA_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIB_PATH:$LD_LIBRARY_PATH"
    echo "NVIDIA Library Path added: $NVIDIA_LIB_PATH"
fi

echo "CRONOS environment setup complete."
