#!/bin/bash
# eval_1x1_shovel_cloth.sh - 1-object / 1-receptacle eval with
# kitchen shovel on cloth via PickPlaceNxM-v1 (N=1, M=1).

# OpenVLA evaluation
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PYTHONPATH:$(pwd) \
python main.py \
    --name "CRONOS_Eval_1x1_shovel_cloth" \
    --seed 0 \
    --env_id "PickPlaceNxM-v1" --env_n 1 --env_m 1 \
    --obj1_index 2 --plate1_index 2 \
    --num_envs 16 \
    --vla_load_path "/mnt/home/ChengDian/workspace/TestCheckpoint/seed0" \
    --eval_sequential \
    --eval_sequences 5 \
    --record_video \
    --no-wandb

# UniVLA evaluation
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PYTHONPATH:$(pwd) \
python main.py \
    --name "CRONOS_UniVLA_Eval_1x1_shovel_cloth" \
    --seed 0 \
    --env_id "PickPlaceNxM-v1" --env_n 1 --env_m 1 \
    --obj1_index 2 --plate1_index 2 \
    --num_envs 16 \
    --vla_type univla \
    --vla_path /mnt/home/guest/Cronos_UniVLA/UniVLA/qwbu__univla-7b-224-sft-simpler-bridge \
    --vla_unnorm_key bridge_oxe \
    --univla_window_size 10 \
    --eval_sequential \
    --eval_sequences 5 \
    --record_video \
    --no-wandb
