#!/bin/bash
# eval.sh - CRONOS standalone evaluation.
#
# Usage: bash scripts/eval.sh <checkpoint_dir> [config] [cuda] [num_eval_episode]
#
# Example:
#   bash scripts/eval.sh .../glob/episode_0128
#   bash scripts/eval.sh .../glob/episode_0128 configs/two_group_sequential_2x2.yaml 2 8
#
# num_envs is read from the config file (per-group num_envs).
#
# Output directory: defaults to a sibling `eval/` of the checkpoint's run, i.e.
# the eval lands inside the same run tree as the checkpoint it evaluates instead
# of wherever the shell happened to be. Override with:
#   RUN_OUT_DIR=/data/runs/my-eval bash scripts/eval.sh .../glob/episode_0128
#
# Previously this script passed no --wandb-dir at all, so eval_only.py fell
# through to wandb's default root — and, depending on the installed wandb
# version, on to $TMPDIR. See run_paths.py for why that was silent.
#
# This script must be run from the CRONOS directory.

set -e

if [ ! -f "eval_only.py" ] || [ ! -d "configs" ]; then
  echo "ERROR: run this from the CRONOS directory (eval_only.py and configs/ must be in \$PWD)."
  echo "       currently in: $(pwd)"
  echo "       try: cd \"$(cd "$(dirname "$0")/.." && pwd)\" && bash scripts/eval.sh $*"
  exit 1
fi

CKPT=${1:?Usage: bash scripts/eval.sh <checkpoint_dir> [config] [cuda] [num_eval_ep]}
CONFIG=${2:-configs/one_group_seq_random_2x2.yaml}
CUDA=${3:-3}
NUM_EVAL_EP=${4:-4}

if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint dir not found: $CKPT"
  exit 1
fi

# --- Run output directory ---
# Default: <run>/eval, where <run> is the wandb run dir that owns the
# checkpoint. $CKPT is .../wandb/run-<ts>-<id>/glob/episode_XXXX, so two levels
# up is the run dir. If $CKPT does not follow that layout (e.g. a hand-placed
# TestCheckpoint/seed0), fall back to ./eval_out under the CWD.
_ckpt_abs="$(cd "$CKPT" && pwd)"
_run_dir="$(dirname "$(dirname "$_ckpt_abs")")"
if [ "$(basename "$(dirname "$_ckpt_abs")")" = "glob" ]; then
  _default_out="$_run_dir/eval"
else
  _default_out="$(pwd)/eval_out"
fi
RUN_OUT_DIR="${RUN_OUT_DIR:-${WANDB_DIR:-$_default_out}}"
case "$RUN_OUT_DIR" in
  /*) ;;
  *)  RUN_OUT_DIR="$(pwd)/$RUN_OUT_DIR" ;;
esac
mkdir -p "$RUN_OUT_DIR"
echo "[eval.sh] checkpoint:     $_ckpt_abs"
echo "[eval.sh] run output dir: $RUN_OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python eval_only.py \
    --name "CRONOS-eval" \
    --seed 0 \
    --env-id PickPlaceNxM-v1 \
    --vla-path openvla/openvla-7b \
    --vla-unnorm-key bridge_orig \
    --config-path "$CONFIG" \
    --segment-len 80 \
    --num-eval-episode $NUM_EVAL_EP \
    --vla-load-path "$CKPT" \
    --wandb-dir "$RUN_OUT_DIR" \
    --record-video
