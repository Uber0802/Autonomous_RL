#!/bin/bash
# Full training with qwbu/univla-7b-224-sft-simpler-bridge (Prismatic backbone)
# Drop-in replacement for OpenVLA — same architecture, same 256-bin action tokenization.
# Only the checkpoint path and normalization key differ from the OpenVLA baseline.
#
# Note: The action_decoder.pt in the SFT checkpoint is for Emu3-based UniVLA only
# and is NOT compatible with the Prismatic backbone. We use standard OpenVLA-style
# bin tokenization instead.

cd SimplerEnv
cuda="0"  # Select GPU

export PYTHONPATH=$(dirname $PWD)/openvla:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="univla-sft-bridge-80-seed-0" \
  --log="univla-sft-bridge-80-seed-0.txt" \
  --wandb_dir=".." \
  --env_id="TwoObjectTwoReceptacle-v1" \
  --vla_path="../checkpoints/univla-7b-sft-bridge" \
  --vla_unnorm_key="bridge_oxe" \
  --seed=0 \
  --training_len=80 \
  --training_interval=80 \
  --max_reset=16384 \
  --max_episodes=256 \
  --interval_eval=16 \
  --interval_save=32

