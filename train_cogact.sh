#!/bin/bash
# Train CogACT with PPO
# Usage: bash train_cogact.sh
#        cuda=1 bash train_cogact.sh                               # use GPU 1
#        bash train_cogact.sh --seed 42                            # pass extra args
#        bash train_cogact.sh --bc_init_path ../bc_checkpoints/gaussian_head_init/bc_init.pt  # with BC warmstart

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

python simpler_env/train_ms3_ppo.py \
    --policy_type cogact \
    --vla_path CogACT/CogACT-Base \
    --vla_unnorm_key bridge_orig \
    --vla_lora_rank 32 \
    --vla_lr 1e-4 \
    --vla_vhlr 3e-3 \
    --env_id TwoObjectTwoReceptacle-v1 \
    --num_envs 64 \
    --episode_len 80 \
    --training_len 80 \
    --training_interval 80 \
    --instruction_switch_interval 80 \
    --max_episodes 256 \
    --max_reset 16384 \
    --interval_eval=16 \
    --interval_save=32 \
    --alg_name ppo \
    --alg_ppo_epoch 1 \
    --alg_gradient_accum 20 \
    --alg_entropy_coef 0.0 \
    --buffer_minibatch 8 \
    --buffer_gamma 0.99 \
    --buffer_lambda 0.95 \
    --seed 0 \
    --name "CogACT-PPO-Test" \
    --wandb_dir ../wandb \
    --log train_cogact.log \
    --bc_init_path ../bc_checkpoints/two_obj_init/bc_init.pt
    "$@"
