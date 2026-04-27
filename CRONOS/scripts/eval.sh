#!/bin/bash
# eval.sh - CRONOS V0.3 standalone evaluation (rotation eval).
#
# Usage: bash scripts/eval.sh <checkpoint_dir> [config] [cuda] [num_eval_episode]
#
# Example:
#   bash scripts/eval.sh .../glob/episode_0128
#   bash scripts/eval.sh .../glob/episode_0128 configs/two_group_sequential_2x2.yaml 2 8
#
# num_envs is read from the config file (per-group num_envs).

set -e

CKPT=${1:?Usage: bash scripts/eval.sh <checkpoint_dir> [config] [cuda] [num_eval_ep]}
CONFIG=${2:-configs/one_group_seq_random_2x2.yaml}
CUDA=${3:-3}
NUM_EVAL_EP=${4:-4}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval_only.py \
    --name "CRONOS-V0.3-eval" \
    --seed 0 \
    --env-id PickPlaceNxM-v1 \
    --vla-path openvla/openvla-7b \
    --vla-unnorm-key bridge_orig \
    --config-path "$CONFIG" \
    --segment-len 80 \
    --num-eval-episode $NUM_EVAL_EP \
    --vla-load-path "$CKPT" \
    --record-video
