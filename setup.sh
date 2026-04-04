#!/bin/bash
set -e  # Exit on error
set -x  # Print commands as they run

# Usage: conda activate cogact && bash setup.sh
# Tested with Python 3.10, CUDA 12.1/12.8

# ============================================
# Step 1: Core dependencies
# ============================================
pip install "setuptools<70.0.0"
pip install "numpy<2.0.0" gymnasium==0.29.1 tyro wandb tqdm transforms3d sapien==3.0.0.b1

# ============================================
# Step 2: PyTorch (CUDA 12.1)
# For CUDA 12.8, use: --index-url https://download.pytorch.org/whl/cu128
# ============================================
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# ============================================
# Step 3: OpenVLA base package (provides prismatic VLM, data loaders)
# CogACT depends on this for the Prismatic backbone
# ============================================
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"
cd openvla && pip install -e . && cd ..

# ============================================
# Step 4: CogACT model code (local package)
# ============================================
cd CogACT && pip install -e . && cd ..

# ============================================
# Step 5: Flash Attention
# Must match your torch version. For torch 2.2 + CUDA 12:
# ============================================
pip install ninja packaging
pip install flash-attn==2.5.5 --no-build-isolation --no-cache-dir
# If compilation fails, uncomment the next line instead (uses prebuilt wheel for torch 2.2 + cu12 + py310):
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# ============================================
# Step 6: Other dependencies
# ============================================
pip install datasets==3.3.2

# ============================================
# Step 7: ManiSkill & SimplerEnv (simulation envs)
# ============================================
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..

# ============================================
# Step 8: Patch prismatic to use ungated Llama-2 mirror
# (Avoids 401 Unauthorized from meta-llama/Llama-2-7b-hf)
# ============================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA2_LOCAL="${SCRIPT_DIR}/llama2_local"
PRISMATIC_LLAMA=$(python -c "import prismatic; import os; print(os.path.join(os.path.dirname(prismatic.__file__), 'models', 'backbones', 'llm', 'llama2.py'))")
if [ -f "$PRISMATIC_LLAMA" ] && [ -d "$LLAMA2_LOCAL" ]; then
    # Use bundled local config+tokenizer — no HF download needed at all
    ESCAPED_PATH=$(echo "$LLAMA2_LOCAL" | sed 's|/|\\/|g')
    sed -i "s|meta-llama/Llama-2-7b-hf|${ESCAPED_PATH}|g" "$PRISMATIC_LLAMA"
    sed -i "s|NousResearch/Llama-2-7b-hf|${ESCAPED_PATH}|g" "$PRISMATIC_LLAMA"
    echo "Patched $PRISMATIC_LLAMA -> local path: $LLAMA2_LOCAL"
elif [ -f "$PRISMATIC_LLAMA" ]; then
    # Fallback: use ungated NousResearch mirror
    sed -i 's|meta-llama/Llama-2-7b-hf|NousResearch/Llama-2-7b-hf|g' "$PRISMATIC_LLAMA"
    echo "Patched $PRISMATIC_LLAMA: meta-llama -> NousResearch (ungated mirror)"
else
    echo "WARNING: Could not find prismatic llama2.py to patch. See FAQ Q1."
fi

echo ""
echo "=== Setup complete ==="
echo "Run: bash train_cogact.sh"

# optional: for ubuntu 2204
# apt-get update
# apt-get install libglvnd-dev
