#!/bin/bash
# setup.sh - Environment setup for CRONOS (+ SpatialVLA integration)
#
# The original CRONOS recipe pinned torch==2.2.0+cu121 / transformers (any).
# The SpatialVLA integration replaces both with newer pins:
#
#   * transformers==4.47.0  — SpatialVLA's model files use APIs from 4.47.0
#                             (GenerationMixin layout, Gemma2 logits format).
#                             Hard ceiling: <4.50 (GenerationMixin removed).
#   * torch==2.7.0+cu128   — supports sm_120 (Blackwell, e.g. RTX PRO 6000).
#                             SpatialVLA's own pyproject pins torch==2.5.1+cu121,
#                             but +cu121 wheels are built for sm_50..sm_90 only,
#                             so generate() throws a "no kernel image" error on
#                             Blackwell. 2.7.0+cu128 is the lowest stable build
#                             with sm_120 kernels.
#
# OpenVLA's setup.py pins are older (torch 2.2.0, transformers 4.40.1, tokenizers
# 0.19.1); pip will print a dep-conflict warning. The conflicts are version-pin
# *declarations*, not runtime API breaks — OpenVLA inference under transformers
# 4.47.0 + torch 2.7.0 is the same code path as cronos-univla (which already
# runs OpenVLA on Blackwell under the same major version family), and the E-0
# gate verifies it end-to-end (`OpenVLA(Prismatic) zero-shot sane`).

set -e

echo "Setting up CRONOS environment..."

# 1. Install core dependencies (Blackwell-compatible)
pip install "setuptools<70.0.0"
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2.0.0" \
    transformers==4.47.0 accelerate==1.0.1 peft==0.14.0 \
    einops==0.8.0 tokenizers==0.21.0 scipy==1.14.1 \
    "timm>=0.9.10,<1.0.0" draccus rich \
    gymnasium==0.29.1 tyro wandb tqdm transforms3d sapien==3.0.0.b1 \
    dacite h5py tabulate trimesh imageio "imageio[ffmpeg]" \
    "mplib==0.1.1" "fast_kinematics==0.2.2" IPython \
    "pytorch_kinematics==0.7.5" pynvml
pip install tensorflow==2.15.0 tensorflow-datasets==4.9.3 tensorflow-graphics
pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# 2. Install backbone packages in editable mode
pip install --no-deps -e ../ManiSkill
pip install --no-deps -e ../SimplerEnv
pip install --no-deps -e ../openvla
# SpatialVLA backbone: vendored as a sibling (no .git inside it) and installed
# editable so the integration code in CRONOS/SimplerEnv can import its
# transformers-style model/processor modules directly. `--no-deps` because the
# core deps above already cover its runtime needs and pinning torch==2.5.1+cu121
# from its pyproject would defeat the Blackwell-compatible torch pin chosen here.
pip install --no-deps -e ../SpatialVLA

echo "CRONOS environment setup complete."
