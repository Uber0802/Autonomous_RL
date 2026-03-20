#!/bin/bash
# eval.sh - Run CRONOS sequential multi-sequence evaluation

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PYTHONPATH:$(pwd) \
/mnt/home/ChengDian/miniconda3/envs/cronos_env/bin/python main.py \
    --name "CRONOS_Eval" \
    --env_id "TwoObjectTwoReceptacle-v1" \
    --vla_load_path "/mnt/home/ChengDian/workspace/TestCheckpoint/seed0" \
    --seed 0 \
    --only_render_seq \
    --eval_sequences 5 \
    --record_video \
    --no_wandb
