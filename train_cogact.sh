#!/bin/bash
# Train CogACT with PPO on PutCarrotOnPlateInScene-v1
# Uses CogACT-Base VLM with Gaussian action head + LoRA

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

conda activate cogact

cd /mnt/home/ChengDian/CRONOS/CogACT_RL

python SimplerEnv/simpler_env/train_ms3_ppo.py \
    --policy_type cogact \
    --vla_path CogACT/CogACT-Base \
    --vla_unnorm_key bridge_orig \
    --vla_lora_rank 32 \
    --vla_lr 1e-4 \
    --vla_vhlr 3e-3 \
    --env_id PutCarrotOnPlateInScene-v1 \
    --num_envs 64 \
    --episode_len 80 \
    --training_len 320 \
    --training_interval 160 \
    --instruction_switch_interval 80 \
    --max_episodes 32 \
    --alg_name ppo \
    --alg_ppo_epoch 1 \
    --alg_gradient_accum 20 \
    --alg_entropy_coef 0.0 \
    --buffer_minibatch 8 \
    --buffer_gamma 0.99 \
    --buffer_lambda 0.95 \
    --seed 0 \
    --name "CogACT-PPO-carrot" \
    --wandb_dir /mnt/home/ChengDian/CRONOS/CogACT_RL/wandb \
    --wandb true \
    --log /mnt/home/ChengDian/CRONOS/CogACT_RL/train_cogact.log \
    "$@"
