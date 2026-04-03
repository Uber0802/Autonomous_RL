#!/bin/bash
set -e  # Exit on error
set -x  # Print commands as they run

# Usage: conda activate cogact && bash setup.sh

# ============================================
# Core PyTorch (CUDA 12.1)
# ============================================
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# ============================================
# OpenVLA base package (provides prismatic VLM, data loaders)
# CogACT depends on this for the Prismatic backbone
# ============================================
cd openvla && pip install -e . && cd ..

# ============================================
# CogACT model code (local package)
# ============================================
cd CogACT && pip install -e . && cd ..

# ============================================
# Flash Attention (prebuilt wheel for CUDA 12 + torch 2.2 + Python 3.10)
# ============================================
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# ============================================
# Other dependencies
# ============================================
pip install -U tyro
pip install datasets==3.3.2

# ============================================
# ManiSkill & SimplerEnv (simulation envs)
# ============================================
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..

# optional: for ubuntu 2204
# apt-get update
# apt-get install libglvnd-dev
