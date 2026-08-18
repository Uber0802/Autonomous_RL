#!/bin/bash
# train.sh - CRONOS training: 3 horizons × 3 segments × 5 reset modes × 2 VLA policies.
#
# Usage: bash scripts/train.sh <mode> [seed] [cuda] [reset] [config] [vla] [eer] [algo] [perturb]
#
#   mode:   t80a|t80b|t80c | t320a|t320b|t320c | t1280a|t1280b|t1280c | t2560a|t2560b|t2560c
#   seed:   random seed (default: 0)
#   cuda:   GPU device (default: 3)
#   reset:  normal|LSR|HSR|LSR+HSR|noep (default: normal)
#   config: YAML experiment config (default: configs/one_group_seq_random_2x2.yaml)
#   vla:    openvla|spatialvla (default: openvla)
#   eer:    on|off — End-Effector Reset (default: on)
#   algo:   ppo|grpo|grpo-scene|grpo-task (default: ppo)
#             ppo         actor-critic + GAE (unchanged)
#             grpo        critic-free, one group per segment (all envs)
#                         (= AutoRL's compute_returns_grpo, bit-identical)
#             grpo-scene  critic-free, one group per (segment, YAML group)
#             grpo-task   critic-free, one group per (segment, fan-out sub-block)
#           Fine-tune the std term with GRPO_STD_SCOPE=group|global|none; see the
#           algorithm block below for the per-mode defaults and why they differ.
#   perturb: off|recep|mixed (default: off) — needs reset ∈ {LSR, LSR+HSR, noep}
#             off    LSR reset goal is always "put X on table" (unchanged)
#             recep  reset goal is another receptacle, != the forward task's
#             mixed  per-env draw between the two; ratio via PERTURB_RECEP_PROB
#
# Reset modes:
#   normal   — standard episodic training (hard reset every episode)
#   LSR      — Low-level State Reset: learn the backward policy
#   HSR      — High-level State Reset: respawn fallen objects every task boundary
#   LSR+HSR  — both backward + reset_unsuitable
#   noep     — LSR+HSR without episodic reset (reset_mode=none)
#
# EER (End-Effector Reset) is orthogonal to all five reset modes: it controls
# whether the gripper is returned to its initial pose at every segment boundary
# (`--reset-robot`, on by default in main.py). Turning it OFF gives a fully
# continuous arm — nothing repositions the end effector between segments.
#   on   — gripper resets every segment, in every reset mode (historical behavior)
#   off  — arm carries its pose across segment boundaries (`--no-reset-robot`)
# With eer=off the run tag gains a `-noEER` suffix so it lands in its own output
# directory; eer=on keeps the existing tag byte-for-byte, so prior runs and
# resume paths are unaffected.
#
# VLA policies (both run in the same conda env after `bash setup.sh all`):
#   openvla    — OpenVLA-7B + bridge_orig unnorm key
#   spatialvla — SpatialVLA-4B-224-SFT-Bridge + bridge_orig/1.0.0
#
# Resume: set CKPT to the previous segment's checkpoint dir:
#   CKPT=.../glob/episode_0128 bash scripts/train.sh t80b 0 3
#   CKPT=.../glob/episode_0032 bash scripts/train.sh t320b 0 3 LSR
#
# Output directory: defaults to ./$RUN_TAG (created before launch, passed to
# main.py as an ABSOLUTE --wandb-dir). Override with:
#   RUN_OUT_DIR=/data/runs/my-run bash scripts/train.sh t320a 0 3
# The run then writes to $RUN_OUT_DIR/wandb/run-<ts>-<id>/{files,glob}. The
# legacy WANDB_DIR=... spelling still works but is discouraged — it is wandb's
# own env var, so exporting it globally collapses every run into one directory.
#
# This script must be run from the CRONOS directory (it uses relative paths for
# --config-path, PYTHONPATH and the default output dir); it checks and exits
# with a clear message otherwise.
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

# --config-path, PYTHONPATH and the default output dir are all relative, so the
# CWD has to be the CRONOS directory. Fail here rather than 20 minutes into a run
# with a FileNotFoundError on the YAML, or with the output dir created in the
# wrong place.
if [ ! -f "main.py" ] || [ ! -d "configs" ]; then
  echo "ERROR: run this from the CRONOS directory (main.py and configs/ must be in \$PWD)."
  echo "       currently in: $(pwd)"
  echo "       try: cd \"$(cd "$(dirname "$0")/.." && pwd)\" && bash scripts/train.sh $*"
  exit 1
fi

MODE=${1:-t80a}
SEED=${2:-0}
CUDA=${3:-3}
RESET=${4:-normal}
CONFIG=${5:-configs/one_group_seq_random_2x2.yaml}
VLA=${6:-openvla}
EER=${7:-on}
ALGO=${8:-ppo}
PERTURB=${9:-off}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# --- VLA → policy + checkpoint + unnorm key ---
case $VLA in
  openvla)
    VLA_TAG="openvla"
    VLA_ARGS="--policy openvla --vla-path openvla/openvla-7b --vla-unnorm-key bridge_orig"
    ;;
  spatialvla)
    # SpatialVLA pinned to the SFT-bridge SFT checkpoint; unnorm key MUST be
    # `bridge_orig/1.0.0` so `get_action_stats` finds the right q01/q99.
    VLA_TAG="spatialvla"
    VLA_ARGS="--policy spatialvla --vla-path IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge --vla-unnorm-key bridge_orig/1.0.0 --vla-temperature-eval 0.0"
    ;;
  *) echo "Unknown vla: $VLA"; echo "Valid: openvla|spatialvla"; exit 1 ;;
esac

ENV_ARGS="--env-id PickPlaceNxM-v1 $VLA_ARGS"

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

# --- EER (End-Effector Reset) → --reset-robot toggle ---
# Orthogonal to $RESET: it decides whether the gripper returns to its initial
# pose at each segment boundary. main.py defaults --reset-robot to true, so "on"
# needs no flag and the emitted command line stays identical to before this
# option existed.
case $EER in
  on)
    EER_TAG=""
    EER_ARGS=""
    ;;
  off)
    # Tag only in the off case, so every existing run directory name, wandb dir
    # and resume path is unchanged for the default.
    EER_TAG="-noEER"
    EER_ARGS="--no-reset-robot"
    ;;
  *) echo "Unknown eer: $EER"; echo "Valid: on|off"; exit 1 ;;
esac

# --- Algorithm (PPO / GRPO) ---
# Orthogonal to every reset knob above. `ppo` emits no flag and no tag, so a
# command line that omits this argument is byte-identical to before the option
# existed — same precedent as EER.
#
# The three GRPO modes differ only in what counts as one group. All three are
# scoped to a single segment, so they form a clean nesting. Sizes in brackets are
# for four_group_sequential_2x2 (64 envs = 4 YAML groups x 16 envs, 4 unique
# tasks per group):
#   grpo        [64] group = one segment, every env. This is exactly what AutoRL
#                    normalizes over — its `train_grpo` runs on a buffer holding
#                    a single segment of num_envs trajectories. Verified
#                    bit-identical, so numbers stay comparable to AutoRL /
#                    RL4VLA GRPO baselines.
#   grpo-scene  [16] group = (segment, YAML group). Same objects, receptacles
#                    and background overlay; tasks may differ within the group.
#   grpo-task   [4]  group = (segment, fan-out sub-block). Only envs that ran
#                    the SAME (object, receptacle) in the SAME segment.
# Group width is `group_num_envs / n_unique_tasks` for task scope, so it depends
# on the config: 4 here, 8 for two_group_sequential_2x2, 16 for the one_group
# configs. main.py prints the actual sizes at startup and warns below 8.
#
# --grpo-std-scope defaults differ per mode on purpose:
#   grpo        -> group  : with one batch-wide group the std is a stable global
#                           scale factor, and it is what AutoRL does.
#   grpo-scene  -> global \  per-group std doubles as a per-group WEIGHT. Once
#   grpo-task   -> global /  the group is a handful of envs that weight is
#                           noise-dominated and biased toward imbalanced groups.
#                           `global` centres per group but scales by the whole
#                           update's std: bias gone, gradient scale unchanged.
# Override any of them with GRPO_STD_SCOPE=group|global|none (tagged when it
# differs from the mode's default so runs land in separate directories).
case $ALGO in
  ppo)
    ALGO_TAG=""
    ALGO_ARGS=""
    _std_default=""
    ;;
  grpo)
    ALGO_TAG="-grpo"
    ALGO_ARGS="--alg-name grpo --grpo-group-scope batch"
    _std_default="group"
    ;;
  grpo-scene)
    ALGO_TAG="-grpoScene"
    ALGO_ARGS="--alg-name grpo --grpo-group-scope scene"
    _std_default="global"
    ;;
  grpo-task)
    ALGO_TAG="-grpoTask"
    ALGO_ARGS="--alg-name grpo --grpo-group-scope task"
    _std_default="global"
    ;;
  *) echo "Unknown algo: $ALGO"; echo "Valid: ppo|grpo|grpo-scene|grpo-task"; exit 1 ;;
esac

# --- Perturbation (arXiv:2004.12570 §4.1), via the LSR reset goal ---
# A reset policy that always drives the object back to the SAME canonical state
# keeps the forward policy's start-state distribution narrow — the exact failure
# the paper identifies. Widening it here means letting the reset segment
# sometimes place the object on a DIFFERENT receptacle instead of on the table.
# Both goals use tasks that already exist in the pool, so there is no new task
# string, no new reward term and no env change: swapping the target receptacle
# makes the env's own `success` predicate and language instruction follow.
#
#   off    always "put X on table" (historical LSR). Default; emits no flag and
#          no tag, and does not even draw from the RNG, so the command line and
#          the numerics are identical to before this option existed.
#   recep  always another receptacle, chosen != the forward task's
#   mixed  per-env, per-reset-segment draw between the two
#
# Requires LSR (--enable-backward), i.e. reset ∈ {LSR, LSR+HSR, noep}; main.py
# errors out otherwise. Envs with only one receptacle fall back to the table
# goal. Tune the mixed ratio with PERTURB_RECEP_PROB (default 0.5).
case $PERTURB in
  off)
    PTB_TAG=""
    PTB_ARGS=""
    ;;
  recep)
    PTB_TAG="-PTBrecep"
    PTB_ARGS="--backward-goal recep"
    ;;
  mixed)
    _p="${PERTURB_RECEP_PROB:-0.5}"
    PTB_TAG="-PTBmixed${_p}"
    PTB_ARGS="--backward-goal mixed --backward-recep-prob $_p"
    ;;
  *) echo "Unknown perturb: $PERTURB"; echo "Valid: off|recep|mixed"; exit 1 ;;
esac

if [ "$PERTURB" != "off" ]; then
  case $RESET in
    LSR|LSR+HSR|noep) ;;
    *) echo "perturb=$PERTURB needs LSR (the reset segment it perturbs)."
       echo "Use reset = LSR | LSR+HSR | noep, got '$RESET'."; exit 1 ;;
  esac
fi

if [ -n "$_std_default" ]; then
  _std="${GRPO_STD_SCOPE:-$_std_default}"
  case $_std in
    group|global|none) ;;
    *) echo "Unknown GRPO_STD_SCOPE: $_std"; echo "Valid: group|global|none"; exit 1 ;;
  esac
  ALGO_ARGS="$ALGO_ARGS --grpo-std-scope $_std"
  [ "$_std" != "$_std_default" ] && ALGO_TAG="${ALGO_TAG}-std${_std}"
fi

# Derive config name from filename (e.g. configs/one_group_sequential_3x3.yaml → one_group_sequential_3x3)
CONFIG_NAME=$(basename "$CONFIG" .yaml)
RUN_TAG="CRONOS-${VLA_TAG}-${CONFIG_NAME}-${HORIZON_TAG}-${RESET_TAG}${EER_TAG}${ALGO_TAG}${PTB_TAG}-seed${SEED}"
CKPT="${CKPT:-}"

# --- Run output directory ---
# Overridable with RUN_OUT_DIR=... (or the legacy WANDB_DIR=..., kept so existing
# launch scripts keep working). Prefer RUN_OUT_DIR: WANDB_DIR is wandb's own
# environment variable, so exporting it globally for one run silently collapses
# every other run into the same directory.
RUN_OUT_DIR="${RUN_OUT_DIR:-${WANDB_DIR:-${RUN_TAG}}}"

# Must be ABSOLUTE and must EXIST before python starts. wandb resolves a
# relative root against the CWD, and — depending on the installed wandb version —
# silently redirects the entire run to $TMPDIR when the directory does not exist
# or is not writable, emitting only a termwarn that is lost in SAPIEN's startup
# output. Everything the run produces (CSVs, checkpoints, videos) then lands in
# /tmp and is gone at the next reboot. `main.py` re-checks this via
# `run_paths.prepare_wandb_dir` / `verify_run_dir`; doing it here too means the
# failure surfaces before a 7B model is loaded.
case "$RUN_OUT_DIR" in
  /*) ;;
  *)  RUN_OUT_DIR="$(pwd)/$RUN_OUT_DIR" ;;
esac
mkdir -p "$RUN_OUT_DIR"
echo "[train.sh] run output dir: $RUN_OUT_DIR"

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

COMMON="python main.py --name \"$RUN_TAG\" --seed $SEED $ENV_ARGS --config-path \"$CONFIG\" --num-eval-episode 4 $RESET_ARGS $EER_ARGS $ALGO_ARGS $PTB_ARGS --record-video --wandb-dir \"$RUN_OUT_DIR\""

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
