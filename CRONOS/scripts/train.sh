#!/bin/bash
# train.sh - CRONOS V0.3 training: 3 horizons × 3 segments × 5 reset modes.
#
# Usage: bash scripts/train.sh <mode> [seed] [cuda] [reset] [config]
#
#   mode:   t80a|t80b|t80c | t320a|t320b|t320c | t1280a|t1280b|t1280c | t2560a|t2560b|t2560c
#   seed:   random seed (default: 0)
#   cuda:   GPU device (default: 3)
#   reset:  normal|LSR|HSR|LSR+HSR|noep (default: normal)
#
# Reset modes:
#   normal   — standard episodic training (hard reset every episode)
#   LSR      — Low-level State Reset: reset_robot between segments
#   HSR      — High-level State Reset: reset_unsuitable between segments
#   LSR+HSR  — both reset_robot + reset_unsuitable
#   noep     — LSR+HSR without episodic reset (reset_mode=none)
#
# Resume: set CKPT to the previous segment's checkpoint dir:
#   CKPT=.../glob/episode_0128 bash scripts/train.sh t80b 0 3
#   CKPT=.../glob/episode_0032 bash scripts/train.sh t320b 0 3 LSR
#
# All values are PER-RUN (relative). max_reset = episodes x 64 (exact for non-HSR,
# ×5 headroom for HSR/LSR+HSR/noep which add soft resets).
#
# ┌──────────┬──────────┬───────────────┬─────────────────┬─────────────┬──────────────┐
# │ Segment  │ Episodes │ Steps (this)  │ Steps (cumul.)  │ Resets (ex) │ Resets (HSR) │
# ├──────────┼──────────┼───────────────┼─────────────────┼─────────────┼──────────────┤
# │ T80  a   │  128     │   655,360     │     655,360     │   8,192     │  40,960      │
# │ T80  b   │  128     │   655,360     │   1,310,720     │   8,192     │  40,960      │
# │ T80  c   │  320     │ 1,638,400     │   2,949,120     │  20,480     │ 102,400      │
# ├──────────┼──────────┼───────────────┼─────────────────┼─────────────┼──────────────┤
# │ T320 a   │   32     │   655,360     │     655,360     │   2,048     │  10,240      │
# │ T320 b   │   32     │   655,360     │   1,310,720     │   2,048     │  10,240      │
# │ T320 c   │   80     │ 1,638,400     │   2,949,120     │   5,120     │  25,600      │
# ├──────────┼──────────┼───────────────┼─────────────────┼─────────────┼──────────────┤
# │ T1280 a  │    8     │   655,360     │     655,360     │     512     │   2,560      │
# │ T1280 b  │    8     │   655,360     │   1,310,720     │     512     │   2,560      │
# │ T1280 c  │   20     │ 1,638,400     │   2,949,120     │   1,280     │   6,400      │
# ├──────────┼──────────┼───────────────┼─────────────────┼─────────────┼──────────────┤
# │ T2560 a  │    4     │   655,360     │     655,360     │     256     │   1,280      │
# │ T2560 b  │    4     │   655,360     │   1,310,720     │     256     │   1,280      │
# │ T2560 c  │   10     │ 1,638,400     │   2,949,120     │     640     │   3,200      │
# └──────────┴──────────┴───────────────┴─────────────────┴─────────────┴──────────────┘

set -e

ENV_ARGS="--env-id PickPlaceNxM-v1 --vla-path openvla/openvla-7b --vla-unnorm-key bridge_orig"

MODE=${1:-t80a}
SEED=${2:-0}
CUDA=${3:-3}
RESET=${4:-normal}
CONFIG=${5:-configs/one_group_seq_random_2x2.yaml}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# --- Horizon tag ---
case $MODE in
  t80a|t80b|t80c)       HORIZON_TAG="T80"   ;;
  t320a|t320b|t320c)     HORIZON_TAG="T320"  ;;
  t1280a|t1280b|t1280c)  HORIZON_TAG="T1280" ;;
  t2560a|t2560b|t2560c)  HORIZON_TAG="T2560" ;;
  *) echo "Unknown mode: $MODE"; echo "Usage: bash scripts/train.sh [t80a|...|t2560c] [seed] [cuda] [reset]"; exit 1 ;;
esac

# --- Reset mode → CLI flags ---
case $RESET in
  normal)
    RESET_TAG="normal"
    RESET_ARGS=""
    ;;
  LSR)
    RESET_TAG="LSR"
    # V0.4 concept fix: LSR = "learn the backward policy", NOT "reset the
    # robot". Backward training has the policy alternate put-X-on-Y with
    # lift-X-off-Y at each task switch (backward_interval=1 ≈ AutoRL paper
    # default). Gripper reset per segment is a separate knob (--reset-robot)
    # that's on by default in main.py Args and stays on across all modes.
    RESET_ARGS="--enable-backward --backward-interval 1"
    ;;
  HSR)
    RESET_TAG="HSR"
    # HSR = High-level State Reset = respawn fallen objects at every task
    # boundary. Orthogonal to LSR (backward learning) and to --reset-robot
    # (gripper reset, default on).
    RESET_ARGS="--reset-unsuitable"
    ;;
  LSR+HSR)
    RESET_TAG="LSR+HSR"
    # LSR (backward learning) + HSR (respawn fallen objects). --reset-robot
    # stays at its Args default (True) so the gripper still resets per segment.
    RESET_ARGS="--enable-backward --backward-interval 1 --reset-unsuitable"
    ;;
  noep)
    RESET_TAG="noep"
    # noep = LSR + HSR + non-episodic continuity (no env.reset between
    # episodes). Same backward + unsuitable-respawn knobs as LSR+HSR, plus
    # --reset-mode none to skip the inter-episode hard reset.
    RESET_ARGS="--enable-backward --backward-interval 1 --reset-unsuitable --reset-mode none"
    ;;
  *) echo "Unknown reset mode: $RESET"; echo "Valid: normal|LSR|HSR|LSR+HSR|noep"; exit 1 ;;
esac

# Derive config name from filename (e.g. configs/one_group_sequential_3x3.yaml → one_group_sequential_3x3)
CONFIG_NAME=$(basename "$CONFIG" .yaml)
RUN_TAG="CRONOS-V0.3-${CONFIG_NAME}-${HORIZON_TAG}-${RESET_TAG}-seed${SEED}"
WANDB_DIR="${WANDB_DIR:-${RUN_TAG}}"
CKPT="${CKPT:-}"

# --- Per-segment max_reset (relative, = max_episodes x num_envs) ---
# Normal/LSR (no HSR): only hard resets at episode boundaries → ep × num_envs.
# HSR/LSR+HSR/noep: HSR can fire at EVERY segment boundary on EVERY env, so the
# worst-case upper bound is max_episodes × (episode_len / segment_len) × num_envs
# — i.e. all segment boundaries × all envs flagged. Earlier code used a flat ×5
# multiplier that was correct for T80 (1 segment / ep) but ~3× too small for
# T1280 (16 segments / ep), causing premature "max_reset exceeded" stops.
# Extract total num_envs from config (sum of per-group num_envs).
_num_envs=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
groups = cfg.get('groups', [])
total = sum(g.get('num_envs', 0) for g in groups)
if total == 0: total = cfg.get('num_envs', 64)
print(total)
")

case $MODE in
  t80a|t80b) _max_ep=128 ;;
  t80c)      _max_ep=320 ;;
  t320a|t320b) _max_ep=32 ;;
  t320c)     _max_ep=80 ;;
  t1280a|t1280b) _max_ep=8 ;;
  t1280c)    _max_ep=20 ;;
  t2560a|t2560b) _max_ep=4 ;;
  t2560c)    _max_ep=10 ;;
esac

# Segments per episode = episode_len / segment_len (segment_len = 80 throughout).
case $HORIZON_TAG in
  T80)   _segs_per_ep=1   ;;
  T320)  _segs_per_ep=4   ;;
  T1280) _segs_per_ep=16  ;;
  T2560) _segs_per_ep=32  ;;
esac

_exact_resets=$(( _max_ep * _num_envs ))
# Worst-case HSR budget: every env flagged at every segment boundary.
_worst_hsr_resets=$(( _max_ep * _segs_per_ep * _num_envs ))

case $RESET in
  HSR|LSR+HSR|noep) MAX_RESET=$_worst_hsr_resets ;;
  *)                MAX_RESET=$_exact_resets ;;
esac

_require_ckpt() {
  if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    echo "CKPT not found: '${CKPT:-<not set>}'"
    echo "Set CKPT=.../glob/episode_XXXX before running ${MODE}."
    exit 1
  fi
}

COMMON="python main.py --name \"$RUN_TAG\" --seed $SEED $ENV_ARGS --config-path \"$CONFIG\" --num-eval-episode 4 $RESET_ARGS --record-video --wandb-dir \"$WANDB_DIR\""

case $MODE in
  # ── T80 ───────────────────────────────────────────────────────────────
  t80a)
    eval $COMMON \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 128 --max-reset $MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 32
    ;;
  t80b)
    _require_ckpt
    eval $COMMON \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 128 --max-reset $MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 32 \
        --vla-load-path "$CKPT"
    ;;
  t80c)
    _require_ckpt
    eval $COMMON \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --max-episodes 320 --max-reset $MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 32 \
        --vla-load-path "$CKPT"
    ;;

  # ── T320 ──────────────────────────────────────────────────────────────
  t320a)
    eval $COMMON \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 32 --max-reset $MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 8
    ;;
  t320b)
    _require_ckpt
    eval $COMMON \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 32 --max-reset $MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 8 \
        --vla-load-path "$CKPT"
    ;;
  t320c)
    _require_ckpt
    eval $COMMON \
        --segment-len 80 --episode-len 320 --task-len 80 --ppo-update-len 160 \
        --max-episodes 80 --max-reset $MAX_RESET \
        --eval-interval 4 --vla-checkpoint-interval 8 \
        --vla-load-path "$CKPT"
    ;;

  # ── T1280 ─────────────────────────────────────────────────────────────
  t1280a)
    eval $COMMON \
        --segment-len 80 --episode-len 1280 --task-len 80 --ppo-update-len 160 \
        --max-episodes 8 --max-reset $MAX_RESET \
        --eval-interval 1 --vla-checkpoint-interval 2
    ;;
  t1280b)
    _require_ckpt
    eval $COMMON \
        --segment-len 80 --episode-len 1280 --task-len 80 --ppo-update-len 160 \
        --max-episodes 8 --max-reset $MAX_RESET \
        --eval-interval 1 --vla-checkpoint-interval 2 \
        --vla-load-path "$CKPT"
    ;;
  t1280c)
    _require_ckpt
    eval $COMMON \
        --segment-len 80 --episode-len 1280 --task-len 80 --ppo-update-len 160 \
        --max-episodes 20 --max-reset $MAX_RESET \
        --eval-interval 1 --vla-checkpoint-interval 2 \
        --vla-load-path "$CKPT"
    ;;

  # ── T2560 ─────────────────────────────────────────────────────────────
  t2560a)
    eval $COMMON \
        --segment-len 80 --episode-len 2560 --task-len 80 --ppo-update-len 160 \
        --max-episodes 4 --max-reset $MAX_RESET \
        --eval-interval 1 --vla-checkpoint-interval 1
    ;;
  t2560b)
    _require_ckpt
    eval $COMMON \
        --segment-len 80 --episode-len 2560 --task-len 80 --ppo-update-len 160 \
        --max-episodes 4 --max-reset $MAX_RESET \
        --eval-interval 1 --vla-checkpoint-interval 1 \
        --vla-load-path "$CKPT"
    ;;
  t2560c)
    _require_ckpt
    eval $COMMON \
        --segment-len 80 --episode-len 2560 --task-len 80 --ppo-update-len 160 \
        --max-episodes 10 --max-reset $MAX_RESET \
        --eval-interval 1 --vla-checkpoint-interval 1 \
        --vla-load-path "$CKPT"
    ;;
esac
