#!/bin/bash
# Full training with qwbu/univla-7b (Prismatic-based, drop-in replacement for OpenVLA)
# Same parameters as the original OpenVLA training, only checkpoint and norm_key changed.

cd SimplerEnv
cuda="0"  # Select GPU

export PYTHONPATH=$(dirname $PWD)/openvla:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="univla-80-seed-0" \
  --log="univla-80-seed-0.txt" \
  --wandb_dir=".." \
  --env_id="TwoObjectTwoReceptacle-v1" \
  --vla_path="../checkpoints/univla-7b" \
  --vla_unnorm_key="bridge_oxe" \
  --seed=0 \
  --training_len=80 \
  --training_interval=80 \
  --max_reset=16384 \
  --max_episodes=256 \
  --interval_eval=16 \
  --interval_save=32

