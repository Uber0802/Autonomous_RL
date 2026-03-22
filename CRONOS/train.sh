#!/bin/bash
# train.sh - Run CRONOS training

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PYTHONPATH:$(pwd) \
/mnt/home/ChengDian/miniconda3/envs/cronos_env/bin/python main.py \
    --name "CRONOS_Train" \
    --seed 0 \
    --env_id "TwoObjectTwoReceptacle-v1" \
    --num_envs 16 \
    --segment_len 80 \
    --episode_len 80 \
    --task_len 80 \
    --max_episodes 1 \
    --no_wandb
