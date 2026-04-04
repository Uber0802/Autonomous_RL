#!/bin/bash
# BC Pretrain: Distill DiT → Gaussian Head
# Usage: bash bc_pretrain.sh
#        cuda=1 bash bc_pretrain.sh

cuda="${cuda:-0}"

cd "$(dirname "$0")/SimplerEnv"

export CUDA_VISIBLE_DEVICES=$cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Ensure libcuda.so is findable (needed by SAPIEN GPU physics)
if ! ldconfig -p 2>/dev/null | grep -q "libcuda.so "; then
    for d in /usr/local/cuda/lib64/stubs /usr/local/cuda/compat /usr/lib/x86_64-linux-gnu; do
        if [ -f "$d/libcuda.so" ] || [ -f "$d/libcuda.so.1" ]; then
            export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
            break
        fi
    done
fi

python simpler_env/bc_pretrain.py \
    --vla_path CogACT/CogACT-Base \
    --vla_unnorm_key bridge_orig \
    --env_id TwoObjectTwoReceptacle-v1 \
    --num_envs 32 \
    --bc_steps 5000 \
    --bc_lr 1e-4 \
    --bc_batch_size 32 \
    --replay_size 20000 \
    --collect_steps 300 \
    --dit_cfg_scale 1.5 \
    --dit_ddim_steps 5 \
    --save_path ../bc_checkpoints/gaussian_head_init \
    --seed 42 \
    "$@"
