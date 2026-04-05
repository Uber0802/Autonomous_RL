#!/bin/bash
# Train with qwbu/univla-7b (Prismatic-based, drop-in replacement for OpenVLA)
# Uses the SAME OpenVLAPolicy code — only checkpoint path and norm_key change.

cd SimplerEnv
cuda="3"  # Select GPU

# PYTHONPATH: include openvla/ for prismatic module
export PYTHONPATH=$(dirname $PWD)/openvla:$PYTHONPATH
# Vulkan ICD for headless servers (needed by SAPIEN/ManiSkill rendering)
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="univla-warmup-seed0" \
  --log="univla-warmup-seed0.txt" \
  --wandb_dir=".." \
  --env_id="TwoObjectTwoReceptacle-v1" \
  --vla_path="../checkpoints/univla-7b" \
  --vla_unnorm_key="bridge_oxe" \
  --seed=0 \
  --num_envs=4 \
  --max_episodes=1 \
  --episode_len=20 \
  --training_len=20 \
  --instruction_switch_interval=20 \
  --training_interval=20 \
  --no-wandb
