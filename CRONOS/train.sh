#!/bin/bash
# train.sh - Run CRONOS training

export CUDA_VISIBLE_DEVICES=0,1

python main.py \
    --name "CRONOS_Baseline" \
    --env_id "TwoObjectOneReceptacle-v1" \
    --num_envs 64 \
    --seed 0 \
    --wandb True
