#!/bin/bash
# test.sh - Twelve training configs for CRONOS (4 horizons x 3 segments) to exercise
# resume_episode restoration across segment splits.
# Usage: bash test.sh [MODE] [seed] [cuda]
#
# Modes: t80a|t80b|t80c | t320a|t320b|t320c
#        univla_t80a|univla_t80b|univla_t80c | univla_t320a|univla_t320b|univla_t320c
#
# num_envs=64, PickPlaceNxM-v1 (N=2, M=2)
#
# ┌──────────────────┬──────────┬───────────────┬─────────────────┬──────────────────┐
# │ Segment          │ Episodes │ Steps (this)  │ Steps (cumul.)  │ Resets (cumul.)  │
# ├──────────────────┼──────────┼───────────────┼─────────────────┼──────────────────┤
# │ T80          a   │  128     │   655,360     │     655,360     │   8,192          │
# │ T80          b   │  128     │   655,360     │   1,310,720     │  16,384          │
# │ T80          c   │  320     │ 1,638,400     │   2,949,120     │  36,864          │
# ├──────────────────┼──────────┼───────────────┼─────────────────┼──────────────────┤
# │ T320         a   │   32     │   655,360     │     655,360     │   2,048          │
# │ T320         b   │   32     │   655,360     │   1,310,720     │   4,096          │
# │ T320         c   │   80     │ 1,638,400     │   2,949,120     │   9,216          │
# ├──────────────────┼──────────┼───────────────┼─────────────────┼──────────────────┤
# │ UniVLA T80   a   │  128     │   655,360     │     655,360     │   8,192          │
# │ UniVLA T80   b   │  128     │   655,360     │   1,310,720     │  16,384          │
# │ UniVLA T80   c   │  320     │ 1,638,400     │   2,949,120     │  36,864          │
# ├──────────────────┼──────────┼───────────────┼─────────────────┼──────────────────┤
# │ UniVLA T320  a   │   32     │   655,360     │     655,360     │   2,048          │
# │ UniVLA T320  b   │   32     │   655,360     │   1,310,720     │   4,096          │
# │ UniVLA T320  c   │   80     │ 1,638,400     │   2,949,120     │   9,216          │
# └──────────────────┴──────────┴───────────────┴─────────────────┴──────────────────┘
#
# Before running *b or *c segments, set the corresponding CKPT_* env var to
# the final checkpoint dir of the previous segment:
#   CKPT_T80=.../glob/episode_0128           bash test.sh t80b          0 3
#   CKPT_T80=.../glob/episode_0256           bash test.sh t80c          0 3
#   CKPT_T320=.../glob/episode_0032          bash test.sh t320b         0 3
#   CKPT_T320=.../glob/episode_0064          bash test.sh t320c         0 3
#   CKPT_UNIVLA_T80=.../glob/episode_0128    bash test.sh univla_t80b   0 3
#   CKPT_UNIVLA_T80=.../glob/episode_0256    bash test.sh univla_t80c   0 3
#   CKPT_UNIVLA_T320=.../glob/episode_0032   bash test.sh univla_t320b  0 3
#   CKPT_UNIVLA_T320=.../glob/episode_0064   bash test.sh univla_t320c  0 3

set -e

ENV_ARGS="--env-id PickPlaceNxM-v1 --env-n 2 --env-m 2 --vla-path openvla/openvla-7b --vla-unnorm-key bridge_orig"
UNIVLA_ARGS="--env-id PickPlaceNxM-v1 --env-n 2 --env-m 2 --vla-type univla --vla-path ../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge --vla-unnorm-key bridge_oxe --univla-window-size 10"

MODE=${1:-t80a}
SEED=${2:-0}
CUDA=${3:-0}

export CUDA_VISIBLE_DEVICES=${CUDA}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PYTHONPATH:$(pwd)"

case $MODE in
  t80a|t80b|t80c)                      HORIZON_TAG="T80"         ;;
  t320a|t320b|t320c)                   HORIZON_TAG="T320"        ;;
  univla_t80a|univla_t80b|univla_t80c) HORIZON_TAG="UniVLA-T80"  ;;
  univla_t320a|univla_t320b|univla_t320c) HORIZON_TAG="UniVLA-T320" ;;
esac
RUN_TAG="CRONOS-V0.2-${HORIZON_TAG}-seed${SEED}"
WANDB_DIR="${WANDB_DIR:-${RUN_TAG}}"
CKPT_T80="${CKPT_T80:-/PATH/TO/t80_prev/glob/episode_XXXX}"
CKPT_T320="${CKPT_T320:-/PATH/TO/t320_prev/glob/episode_XXXX}"
CKPT_UNIVLA_T80="${CKPT_UNIVLA_T80:-/PATH/TO/univla_t80_prev/glob/episode_XXXX}"
CKPT_UNIVLA_T320="${CKPT_UNIVLA_T320:-/PATH/TO/univla_t320_prev/glob/episode_XXXX}"

# max_reset covers full a+b+c cumulative resets with headroom for soft resets
T80_MAX_RESET=40960      # 576 ep * 64 = 36,864 hard + headroom
T320_MAX_RESET=16384     # 144 ep * 64 =  9,216 hard + headroom

_require_ckpt() {
  local ckpt="$1" label="$2"
  if [ ! -d "$ckpt" ]; then
    echo "${label} not found: $ckpt"
    echo "Set ${label}=... or edit test.sh before running ${MODE}."
    exit 1
  fi
}

case $MODE in
  # ── OpenVLA T80 ──────────────────────────────────────────────────────────
  t80a)
    # 128 episodes | 0 → 655,360 steps
    python main.py \
        --name "$RUN_TAG" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 128 --max-reset $T80_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 32 \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  t80b)
    # 128 episodes | 655,360 → 1,310,720 steps, resumes from t80a
    _require_ckpt "$CKPT_T80" "CKPT_T80"
    python main.py \
        --name "${RUN_TAG}-b" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 128 --max-reset $T80_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 32 \
        --vla-load-path "$CKPT_T80" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  t80c)
    # 320 episodes | 1,310,720 → 2,949,120 steps, resumes from t80b
    _require_ckpt "$CKPT_T80" "CKPT_T80"
    python main.py \
        --name "${RUN_TAG}-c" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 320 --max-reset $T80_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 32 \
        --vla-load-path "$CKPT_T80" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;

  # ── OpenVLA T320 ─────────────────────────────────────────────────────────
  t320a)
    # 32 episodes | 0 → 655,360 steps
    python main.py \
        --name "$RUN_TAG" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 32 --max-reset $T320_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 8 \
        --reset-robot \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  t320b)
    # 32 episodes | 655,360 → 1,310,720 steps, resumes from t320a
    _require_ckpt "$CKPT_T320" "CKPT_T320"
    python main.py \
        --name "${RUN_TAG}-b" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 32 --max-reset $T320_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 8 \
        --reset-robot \
        --vla-load-path "$CKPT_T320" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  t320c)
    # 80 episodes | 1,310,720 → 2,949,120 steps, resumes from t320b
    _require_ckpt "$CKPT_T320" "CKPT_T320"
    python main.py \
        --name "${RUN_TAG}-c" --seed $SEED $ENV_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 80 --max-reset $T320_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 8 \
        --reset-robot \
        --vla-load-path "$CKPT_T320" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;

  # ── UniVLA T80 ───────────────────────────────────────────────────────────
  univla_t80a)
    # 128 episodes | 0 → 655,360 steps
    python main.py \
        --name "$RUN_TAG" --seed $SEED $UNIVLA_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 128 --max-reset $T80_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 32 \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  univla_t80b)
    # 128 episodes | 655,360 → 1,310,720 steps, resumes from univla_t80a
    _require_ckpt "$CKPT_UNIVLA_T80" "CKPT_UNIVLA_T80"
    python main.py \
        --name "${RUN_TAG}-b" --seed $SEED $UNIVLA_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 128 --max-reset $T80_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 32 \
        --vla-load-path "$CKPT_UNIVLA_T80" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  univla_t80c)
    # 320 episodes | 1,310,720 → 2,949,120 steps, resumes from univla_t80b
    _require_ckpt "$CKPT_UNIVLA_T80" "CKPT_UNIVLA_T80"
    python main.py \
        --name "${RUN_TAG}-c" --seed $SEED $UNIVLA_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 320 --max-reset $T80_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 32 \
        --vla-load-path "$CKPT_UNIVLA_T80" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;

  # ── UniVLA T320 ──────────────────────────────────────────────────────────
  univla_t320a)
    # 32 episodes | 0 → 655,360 steps
    python main.py \
        --name "$RUN_TAG" --seed $SEED $UNIVLA_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 32 --max-reset $T320_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 8 \
        --reset-robot \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  univla_t320b)
    # 32 episodes | 655,360 → 1,310,720 steps, resumes from univla_t320a
    _require_ckpt "$CKPT_UNIVLA_T320" "CKPT_UNIVLA_T320"
    python main.py \
        --name "${RUN_TAG}-b" --seed $SEED $UNIVLA_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 32 --max-reset $T320_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 8 \
        --reset-robot \
        --vla-load-path "$CKPT_UNIVLA_T320" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;
  univla_t320c)
    # 80 episodes | 1,310,720 → 2,949,120 steps, resumes from univla_t320b
    _require_ckpt "$CKPT_UNIVLA_T320" "CKPT_UNIVLA_T320"
    python main.py \
        --name "${RUN_TAG}-c" --seed $SEED $UNIVLA_ARGS --num-envs 64 \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 80 --max-reset $T320_MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 8 \
        --reset-robot \
        --vla-load-path "$CKPT_UNIVLA_T320" \
        --record-video --wandb-dir "$WANDB_DIR"
    ;;

  *)
    echo "Usage: bash test.sh [t80a|t80b|t80c|t320a|t320b|t320c|univla_t80a|univla_t80b|univla_t80c|univla_t320a|univla_t320b|univla_t320c] [seed] [cuda]"
    exit 1
    ;;
esac
