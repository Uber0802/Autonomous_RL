#!/bin/bash
# eval_start_scenes.sh - start-scene sequential evaluation of one checkpoint.
#
# Runs eval_only.py once per start scene (evaluation/start_scenes.py). Every
# scene of every checkpoint uses the same --seed, so start poses and
# perturbations are identical across checkpoints and the per-trial CSVs pair up.
#
# Usage: bash scripts/eval_start_scenes.sh <ckpt_dir> <tag> [cuda] [rounds] [scenes] [schedule]
#   ckpt_dir   the episode_XXXX/ directory of a training run
#   tag        short label for the output folder (e.g. T80_ep576)
#   cuda       GPU id (default 0)
#   rounds     --eval-rounds (default 0 = the training round of pose set 0)
#   scenes     space-separated start scenes (default: the three overlap scenes)
#   schedule   parallel (default: every group on its own envs at once) | serial
#
# Environment: CONFIG (default: the checkpoint's training config), SEED (0),
# MODE (sequential | single), DOMAINS (in_domain), PY (python).
#
# Outputs: eval_runs/<tag>/<scene>/wandb/offline-run-*/glob/{eval_per_trial.csv,
#          eval_report.txt,init_scenes/*.png,eval_videos/}
#
# Run from the CRONOS directory.

set -e

CKPT=${1:?Usage: bash scripts/eval_start_scenes.sh <ckpt_dir> <tag> [cuda] [rounds] [scenes] [schedule]}
TAG=${2:?missing <tag>}
CUDA=${3:-0}
ROUNDS=${4:-0}
SCENES=${5:-"overlap_obj_nontarget overlap_nonobj_target overlap_nonobj_nontarget"}
SCHEDULE=${6:-parallel}
SEED=${SEED:-0}
MODE=${MODE:-sequential}   # sequential: one reset per round of tasks; single: reset before every task
DOMAINS=${DOMAINS:-in_domain}
PY=${PY:-python}
CONFIG_ARGS=()
[ -n "${CONFIG:-}" ] && CONFIG_ARGS=(--config-path "${CONFIG}")

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_SILENT=true

for SCENE in $SCENES; do
  OUT="eval_runs/${TAG}/${SCENE}"
  if ls "${OUT}"/wandb/offline-run-*/glob/eval_report.txt >/dev/null 2>&1; then
    echo "[skip] ${OUT} already has an eval_report.txt"
    continue
  fi
  mkdir -p "${OUT}"
  echo "=== ${TAG} / ${SCENE} ($(date)) ==="
  "${PY}" eval_only.py \
    --name "CRONOS-eval-${TAG}-${SCENE}" \
    --seed "${SEED}" \
    --vla-load-path "${CKPT}" \
    "${CONFIG_ARGS[@]}" \
    --segment-len 80 \
    --eval-mode "${MODE}" \
    --eval-rounds "${ROUNDS}" \
    --eval-domains "${DOMAINS}" \
    --eval-scene-schedule "${SCHEDULE}" \
    --start-scene "${SCENE}" \
    --record-video --video-envs-per-block 1 \
    --wandb-dir "${OUT}" \
    2>&1 | grep -vE "Warning|warn\(|pynvml|WARNING|HybridCache|do_sample"
done
echo "=== ${TAG} done ($(date)) ==="
