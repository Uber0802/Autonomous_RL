#!/bin/bash
set -e  # Exit on error
set -x  # Print commands as they run

# =============================================================================
# UniVLA_RL Setup — aligned with AutoRL's install flow
#
# AutoRL order: torch → openvla (pip -e) → tyro → datasets → flash-attn → ManiSkill → SimplerEnv
# We follow the same order, with CUDA auto-detection added.
# =============================================================================

# ===== Step 0: Detect CUDA version =====
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "12.1")
CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VER | cut -d. -f2)
echo "Detected CUDA $CUDA_VER"

# ===== Step 1: Install PyTorch (same as AutoRL line 5) =====
if [ "$CUDA_MAJOR" -ge 13 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]); then
    echo "===== CUDA >= 12.8 detected — using torch 2.7.0 + cu128 ====="
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
    echo "===== CUDA < 12.8 detected — using torch 2.2.0 + cu121 ====="
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
fi

# ===== Step 2: Install openvla (same as AutoRL line 6) =====
# This pulls transformers==4.40.1, peft==0.11.1, tensorflow, etc. via pyproject.toml
# NOTE: pyproject.toml has been relaxed to torch>=2.2.0 to support cu128
cd openvla && pip install -e . && cd ..

# ===== Step 3: Upgrade tyro (same as AutoRL line 7) =====
pip install -U tyro

# ===== Step 4: Install datasets (same as AutoRL line 8) =====
pip install datasets==3.3.2

# ===== Step 5: Install flash-attn (same as AutoRL lines 10-13) =====
# CRITICAL: flash-attn must be compiled against the SAME torch version.
#   - cu121 + torch 2.2.0 → use prebuilt wheel (fast)
#   - cu128 + torch 2.7.0 → build from source (slow, ~10 min)
# If you see "undefined symbol: _ZNK3c105Error4whatEv", flash-attn was built
# against a different torch. Fix: pip uninstall flash-attn && rebuild.
if [ "$CUDA_MAJOR" -ge 13 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]); then
    pip install psutil  # required by flash-attn build
    pip install flash-attn --no-build-isolation --no-cache-dir
else
    wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi

# ===== Step 6: Install ManiSkill & SimplerEnv (same as AutoRL lines 16-17) =====
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..

# ===== Step 7: Download checkpoint + patch (UniVLA-specific) =====
echo "===== Downloading qwbu/univla-7b checkpoint (~14 GB) ====="
huggingface-cli download qwbu/univla-7b --local-dir checkpoints/univla-7b
python patch_checkpoint.py

# ===== Step 8: Download ManiSkill assets =====
python -m mani_skill.utils.download_asset bridge_v2_real2sim

# ===== Step 9: Verify =====
echo "===== Verifying installation ====="
PYTHONPATH=$PWD/openvla:$PYTHONPATH python -c "
import torch; print(f'torch: {torch.__version__}, CUDA: {torch.version.cuda}')
import transformers; print(f'transformers: {transformers.__version__}')
import flash_attn; print(f'flash_attn: {flash_attn.__version__}')
import peft; print(f'peft: {peft.__version__}')
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead; print('prismatic: OK')
import mani_skill; print('mani_skill: OK')
print('All imports passed!')
"

echo "===== Setup complete ====="
echo "Next: bash train_univla_full.sh"
