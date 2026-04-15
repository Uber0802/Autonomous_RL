#!/bin/bash
# setup.sh - Environment setup for CRONOS + UniVLA

set -e

# Anchor to the script's own directory so the ../X relative paths below resolve
# correctly regardless of the caller's working directory.
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Setting up CRONOS + UniVLA environment..."

# 1. Install core dependencies
pip install "setuptools<70.0.0"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0.0" gymnasium==0.29.1 tyro wandb tqdm transforms3d sapien==3.0.0.b1
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# 2. Install backbone packages in editable mode
pip install -e ../ManiSkill
pip install -e ../SimplerEnv
pip install -e ../openvla
pip install -e ../UniVLA      # makes 'prismatic' from UniVLA importable

# 3. UniVLA-specific additional deps
pip install einops             # required by ActionDecoderHead (MAPBlock uses einops)
# transforms3d already installed above
# Note: tensorflow IS installed transitively (dlimp and openvla both require it).
# univla_action_decoder.py (Phase 2) removes TF from UniVLA's own code path — it is
# not used by the CRONOS runner — but TF remains present as a dlimp/openvla dep.

echo "Setup complete."
