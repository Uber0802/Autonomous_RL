#!/bin/bash
# train.sh - CRONOS training launch scripts
# Usage: bash train.sh [80|320|1280]

set -e

PYTHON=/mnt/home/ChengDian/miniconda3/envs/cronos_env/bin/python
COMMON="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
ENV_ARGS="--env_id PickPlaceNxM-v1 --env_n 2 --env_m 2 --vla_path openvla/openvla-7b --vla_unnorm_key bridge_orig"

MODE=${1:-80}

case $MODE in
  80)
    # T=80: segment_len=80, episode_len=80, ppo_update_len=80
    # eval every 4 episodes, checkpoint every 32 episodes
    eval $COMMON PYTHONPATH=\$PYTHONPATH:\$(pwd) $PYTHON main.py \
        --name "CRONOS-T80" --seed 0 $ENV_ARGS --num_envs 16 \
        --segment_len 80 --episode_len 80 --task_len 80 --ppo_update_len 80 \
        --max_episodes 32 --eval_interval 4 --vla_checkpoint_interval 32
    ;;
  320)
    # T=320: segment_len=80, episode_len=320, ppo_update_len=160
    # eval every 4 episodes, checkpoint every 8 episodes, reset_robot
    eval $COMMON PYTHONPATH=\$PYTHONPATH:\$(pwd) $PYTHON main.py \
        --name "CRONOS-T320" --seed 0 $ENV_ARGS --num_envs 16 \
        --segment_len 80 --episode_len 320 --task_len 80 --ppo_update_len 160 \
        --max_episodes 32 --eval_interval 4 --vla_checkpoint_interval 8 \
        --reset_robot
    ;;
  1280)
    # T=1280: segment_len=80, episode_len=1280, ppo_update_len=160
    # eval every 1 episode, checkpoint every 2 episodes, reset_robot
    eval $COMMON PYTHONPATH=\$PYTHONPATH:\$(pwd) $PYTHON main.py \
        --name "CRONOS-T1280" --seed 0 $ENV_ARGS --num_envs 16 \
        --segment_len 80 --episode_len 1280 --task_len 80 --ppo_update_len 160 \
        --max_episodes 32 --eval_interval 1 --vla_checkpoint_interval 2 \
        --reset_robot
    ;;
  *)
    echo "Usage: bash train.sh [80|320|1280]"
    exit 1
    ;;
esac
