#!/bin/bash
# train.sh - Run CRONOS training

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PYTHONPATH:$(pwd) \
/mnt/home/ChengDian/miniconda3/envs/cronos_env/bin/python main.py \
    --name "CRONOS_Align_V9" \
    --env_id "TwoObjectTwoReceptacle-v1" \
    --num_envs 16 \
    --training_len 80 \
    --training_interval 80 \
    --instruction_switch_interval 80 \
    --max_episodes 1 \
    --seed 0 \
    --no_wandb
