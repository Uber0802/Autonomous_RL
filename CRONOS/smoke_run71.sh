#!/usr/bin/env bash
# Step 7.1 — Minimal UniVLA Training Run
# Run this on the GPU machine (172.16.186.4):
#   bash /mnt/lab-home/guest/Integration/cronos_univla/CRONOS/smoke_run71.sh

set -euo pipefail

# ── Paths (as seen from the GPU machine) ────────────────────────────────────
CRONOS_DIR="/mnt/lab-home/guest/Integration/cronos_univla/CRONOS"
LOG_DIR="$CRONOS_DIR/smoke_test_logs"
LOG_FILE="$LOG_DIR/step71.log"

# ── Config ───────────────────────────────────────────────────────────────────
SMOKE_ENV="cronos-univla-smoke"
SMOKE_GPU=3

mkdir -p "$LOG_DIR"

echo "========================================"
echo " CRONOS × UniVLA — Step 7.1 Smoke Run"
echo " $(date)"
echo " GPU: $SMOKE_GPU   ENV: $SMOKE_ENV"
echo " Log: $LOG_FILE"
echo "========================================"

cd "$CRONOS_DIR"
unset PYTHONPATH

conda run -n "$SMOKE_ENV" --no-capture-output \
  env \
    CUDA_VISIBLE_DEVICES=$SMOKE_GPU \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$CRONOS_DIR" \
  python main.py \
    --name "smoke-univla" \
    --seed 0 \
    --env_id PickPlaceNxM-v1 --env_n 2 --env_m 2 \
    --vla_type univla \
    --vla_path ../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge \
    --vla_unnorm_key bridge_oxe \
    --univla_window_size 10 \
    --num_envs 2 \
    --segment_len 16 --episode_len 16 --task_len 16 --ppo_update_len 16 \
    --max_episodes 1 \
    --debug_rollout \
    --no-wandb \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
echo " Run finished — exit code: $EXIT_CODE"
echo " Log saved to: $LOG_FILE"
echo "========================================"

exit $EXIT_CODE
