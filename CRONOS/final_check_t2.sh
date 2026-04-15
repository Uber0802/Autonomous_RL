#!/usr/bin/env bash
# T-2: Full training regression — UniVLA end-to-end (final_check.md)
# Tests the eval() reset_action_decoders() fix + full training pipeline.
# Run on the GPU machine (172.16.186.4):
#   bash /mnt/lab-home/guest/Integration/cronos_univla/CRONOS/final_check_t2.sh

set -euo pipefail

# ── Paths (as seen from the GPU machine) ────────────────────────────────────
CRONOS_DIR="/mnt/lab-home/guest/Integration/cronos_univla/CRONOS"
LOG_DIR="$CRONOS_DIR/smoke_test_logs"
LOG_FILE="$LOG_DIR/final_check_t2.log"

# ── Config ───────────────────────────────────────────────────────────────────
SMOKE_ENV="cronos-univla-smoke"
SMOKE_GPU=3

mkdir -p "$LOG_DIR"

echo "========================================"
echo " CRONOS × UniVLA — Final Check T-2"
echo " $(date)"
echo " GPU: $SMOKE_GPU   ENV: $SMOKE_ENV"
echo " Log: $LOG_FILE"
echo "========================================"

cd "$CRONOS_DIR"
unset PYTHONPATH

# --eval_interval 1 forces in-training eval after the single episode,
# exercising eval() across all tasks and proving the reset fix works in-context.
conda run -n "$SMOKE_ENV" --no-capture-output \
  env \
    CUDA_VISIBLE_DEVICES=$SMOKE_GPU \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$CRONOS_DIR" \
  python main.py \
    --name "final-check-univla" \
    --seed 0 \
    --env_id PickPlaceNxM-v1 --env_n 2 --env_m 2 \
    --vla_type univla \
    --vla_path ../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge \
    --vla_unnorm_key bridge_oxe \
    --univla_window_size 10 \
    --num_envs 2 \
    --segment_len 16 --episode_len 16 --task_len 16 --ppo_update_len 16 \
    --max_episodes 1 \
    --eval_interval 1 \
    --debug_rollout \
    --no-wandb \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
echo " Run finished — exit code: $EXIT_CODE"
echo " Log saved to: $LOG_FILE"
echo "========================================"

# ── Auto-verify success criteria ─────────────────────────────────────────────
echo ""
echo "Verifying success criteria..."
FAIL=0

grep -q "\[INIT\]" "$LOG_FILE" \
  && echo "[OK] [INIT] line present" \
  || { echo "[FAIL] [INIT] line missing"; FAIL=1; }

grep -q "\[PPO LOSS\]" "$LOG_FILE" \
  && echo "[OK] [PPO LOSS] line present" \
  || { echo "[FAIL] [PPO LOSS] line missing"; FAIL=1; }

EVAL_COUNT=$(grep -c "^Evaluating:" "$LOG_FILE" || true)
if [ "$EVAL_COUNT" -ge 4 ]; then
  echo "[OK] in-training eval ran for $EVAL_COUNT tasks"
else
  echo "[FAIL] expected >=4 'Evaluating:' lines, got $EVAL_COUNT"
  FAIL=1
fi

if grep -qiE "ImportError|ModuleNotFoundError|AttributeError|Traceback" "$LOG_FILE"; then
  echo "[FAIL] errors detected in log — check $LOG_FILE"
  FAIL=1
else
  echo "[OK] no errors in log"
fi

grep -q "Training complete:" "$LOG_FILE" \
  && echo "[OK] clean exit (Training complete)" \
  || { echo "[FAIL] 'Training complete:' not found — run may have crashed"; FAIL=1; }

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "=== T-2 PASSED ==="
else
  echo "=== T-2 FAILED — see failures above and check $LOG_FILE ==="
  exit 1
fi

exit "$EXIT_CODE"
