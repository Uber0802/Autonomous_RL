#!/bin/bash
set -e  # Exit on error
set -x  # Print commands as they run

# ===== Base Dependencies =====
pip install "setuptools<70.0.0"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# ===== Core Dependencies =====
# transformers 4.40.1 required by OpenVLA/Prismatic (PrismaticProcessor.from_pretrained)
# Note: 4.44.0 works for Emu3-based UniVLA but breaks PrismaticProcessor auto-loading
pip install transformers==4.40.1
pip install tiktoken==0.6.0
pip install pillow
pip install accelerate>=0.25.0
pip install peft==0.11.1
pip install sentencepiece==0.1.99

# ===== AutoRL / RL4VLA Dependencies =====
pip install "numpy<2.0.0" gymnasium==0.29.1 tyro wandb tqdm transforms3d sapien==3.0.0.b1
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"
pip install einops timm==0.9.10 rich matplotlib
pip install datasets==3.3.2
pip install huggingface_hub jsonlines json-numpy protobuf draccus==0.8.0

# ===== Flash Attention =====
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# ===== Install local packages (editable) =====
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..

# ===== Note: openvla/prismatic is loaded via PYTHONPATH, no pip install needed =====
# The openvla/ directory provides prismatic module for qwbu/univla-7b (same architecture)
# Set PYTHONPATH=$PWD/openvla:$PYTHONPATH before running training scripts

# ===== Download Checkpoints (uncomment to download) =====
# huggingface-cli download qwbu/univla-7b --local-dir checkpoints/univla-7b
# huggingface-cli download qwbu/univla-7b-224-sft-simpler-bridge --local-dir checkpoints/univla-7b-sft-bridge

# Torchvision Installation
# (Optional) For some machines with CUDA 12.8:
# pip uninstall -y torch torchvision torchaudio flash-attn
# pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# pip install flash-attn --no-build-isolation --no-cache-dir
# Torchvision Installation

echo "===== Setup complete ====="
