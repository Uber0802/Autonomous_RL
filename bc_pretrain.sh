#!/bin/bash
# BC Pretrain: Distill DiT → Gaussian Head
# Usage: bash bc_pretrain.sh
#        cuda=1 bash bc_pretrain.sh

cuda="${cuda:-0}"

cd "$(dirname "$0")/SimplerEnv"

export CUDA_VISIBLE_DEVICES=$cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python simpler_env/bc_pretrain.py \
    --vla_path CogACT/CogACT-Base \
    --vla_unnorm_key bridge_orig \
    --env_id PutCarrotOnPlateInScene-v1 \
    --num_envs 16 \
    --bc_steps 2000 \
    --bc_lr 3e-4 \
    --bc_batch_size 16 \
    --replay_size 5000 \
    --collect_steps 50 \
    --dit_cfg_scale 1.5 \
    --dit_ddim_steps 5 \
    --save_path ../bc_checkpoints/gaussian_head_init \
    --seed 42 \
    "$@"
