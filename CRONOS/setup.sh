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

echo "CRONOS environment setup complete."
