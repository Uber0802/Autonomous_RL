#!/bin/bash
# test.sh - Four training configs for CRONOS (2 horizons x 2 halves) to exercise
# resume_episode restoration across the halfway split.
# Usage: bash test.sh [t80a|t80b|t320a|t320b] [seed] [cuda]
#
# num_envs=64, PickPlaceNxM-v1 (N=2, M=2)
#   t80a : episode_len=80,   steps       0 -> 655360   (128 ep * 80  * 64)
#   t80b : episode_len=80,   steps  655360 -> 1310720  (128 ep * 80  * 64), resumes from t80a
#   t320a: episode_len=320,  steps       0 -> 655360   ( 32 ep * 320 * 64)
#   t320b: episode_len=320,  steps  655360 -> 1310720  ( 32 ep * 320 * 64), resumes from t320a
#
# Before running the *b halves, set CKPT_T80 / CKPT_T320 to the final checkpoint dir
# of the matching *a run (e.g. .../wandb/run-XXXX/glob/episode_0128 for t80a,
# .../glob/episode_0032 for t320a).
# WANDB_DIR is a placeholder — replace with the desired wandb root.

set -e

# N=2, M=2 matches train.sh. Objects/plates are determined by the NxM shape
# spec (DEFAULT_OBJ_INDICES=[7,2], DEFAULT_PLATE_INDICES=[1,2]) and the
# episode_id-driven Lehmer code — no per-run index args needed.
ENV_ARGS="--env-id PickPlaceNxM-v1 --env-n 2 --env-m 2 --vla-path openvla/openvla-7b --vla-unnorm-key bridge_orig"

MODE=${1:-t80a}
SEED=${2:-0}
CUDA=${3:-0}

export CUDA_VISIBLE_DEVICES=${CUDA}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# wandb dir defaults to a mode+seed-specific folder; override via env var if needed.
case $MODE in
  t80a)  HORIZON_TAG="T80"  ;;
  t80b)  HORIZON_TAG="T80"  ;;
  t320a) HORIZON_TAG="T320" ;;
  t320b) HORIZON_TAG="T320" ;;
esac
RUN_TAG="CRONOS-V0.1-${HORIZON_TAG}-seed${SEED}"
WANDB_DIR="${WANDB_DIR:-${RUN_TAG}}"                          # <-- replace if desired
CKPT_T80="${CKPT_T80:-/PATH/TO/t80a/glob/episode_0128}"       # <-- replace before t80b
CKPT_T320="${CKPT_T320:-/PATH/TO/t320a/glob/episode_0032}"    # <-- replace before t320b

case $MODE in
  t80a)
    # EP_len=80 | 128 episodes -> 655360 env steps (64 * 80 * 128)
    python main.py \
        --name "$RUN_TAG" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 128 --eval-interval 4 --vla-checkpoint-interval 32 \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  t80b)
    # EP_len=80 | 128 episodes -> +655360 (cumulative 1310720), resumes from t80a
    if [ ! -d "$CKPT_T80" ]; then
      echo "CKPT_T80 not found: $CKPT_T80"
      echo "Set CKPT_T80=... or edit test.sh before running t80b."
      exit 1
    fi
    python main.py \
        --name "${RUN_TAG}-cont" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 128 --eval-interval 4 --vla-checkpoint-interval 32 \
        --vla-load-path "$CKPT_T80" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  t320a)
    # EP_len=320 | 32 episodes -> 655360 env steps (64 * 320 * 32)
    python main.py \
        --name "$RUN_TAG" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 32 --eval-interval 4 --vla-checkpoint-interval 8 \
        --reset-robot \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  t320b)
    # EP_len=320 | 32 episodes -> +655360 (cumulative 1310720), resumes from t320a
    if [ ! -d "$CKPT_T320" ]; then
      echo "CKPT_T320 not found: $CKPT_T320"
      echo "Set CKPT_T320=... or edit test.sh before running t320b."
      exit 1
    fi
    python main.py \
        --name "${RUN_TAG}-cont" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 32 --eval-interval 4 --vla-checkpoint-interval 8 \
        --reset-robot \
        --vla-load-path "$CKPT_T320" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  *)
    echo "Usage: bash test.sh [t80a|t80b|t320a|t320b] [seed] [cuda]"
    exit 1
    ;;
esac
