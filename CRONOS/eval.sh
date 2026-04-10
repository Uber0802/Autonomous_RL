#!/bin/bash
# eval.sh - Run CRONOS sequential multi-sequence evaluation

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PYTHONPATH:$(pwd) \
/mnt/home/ChengDian/miniconda3/envs/cronos_env/bin/python main.py \
    --name "CRONOS_Eval" \
    --seed 0 \
    --env_id "PickPlaceNxM-v1" --env_n 2 --env_m 2 \
    --num_envs 16 \
    --vla_load_path "/mnt/home/ChengDian/workspace/TestCheckpoint/seed0" \
    --eval_sequential \
    --eval_sequences 5 \
    --record_video \
    --no_wandb
