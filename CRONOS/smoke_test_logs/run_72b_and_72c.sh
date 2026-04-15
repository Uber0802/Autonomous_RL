#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# smoke_test_logs/run_72b_and_72c.sh
#
# Runs step 7.2-B (eval) then all four 7.2-C verification checks.
# Output is saved to:
#   smoke_test_logs/step72_rerun.log   — full eval transcript
#   smoke_test_logs/step72c_verify.log — 7.2-C check results
#
# Usage (on 172.16.186.4):
#   cd /mnt/home/guest/Integration/cronos_univla/CRONOS
#   bash smoke_test_logs/run_72b_and_72c.sh
# ---------------------------------------------------------------------------
set -euo pipefail

export SMOKE_ENV=cronos-univla-smoke
export SMOKE_GPU=3
export CUDA_VISIBLE_DEVICES=$SMOKE_GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CRONOS_DIR=/mnt/home/guest/Integration/cronos_univla/CRONOS
LOG_DIR=$CRONOS_DIR/smoke_test_logs

cd "$CRONOS_DIR"

# ---------- retrieve checkpoint path ----------
CKPT_PATH=$(cat "$LOG_DIR/checkpoint_path.txt")
echo "[INFO] Checkpoint: $CKPT_PATH"
ls "$CKPT_PATH" || { echo "[ERROR] Checkpoint directory not found"; exit 1; }

# ---------- 7.2-B: run eval ----------
echo ""
echo "============================================================"
echo " 7.2-B  Running eval — output -> step72_rerun.log"
echo "============================================================"

unset PYTHONPATH
conda run -n "$SMOKE_ENV" --no-capture-output \
  env \
    CUDA_VISIBLE_DEVICES=$SMOKE_GPU \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$CRONOS_DIR" \
  python main.py \
    --name "smoke-univla-eval-rerun" \
    --seed 0 \
    --env_id PickPlaceNxM-v1 --env_n 2 --env_m 2 \
    --vla_type univla \
    --vla_path ../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge \
    --vla_unnorm_key bridge_oxe \
    --univla_window_size 10 \
    --num_envs 2 \
    --segment_len 16 --episode_len 16 --task_len 16 \
    --vla_load_path "$CKPT_PATH" \
    --eval_sequential \
    --eval_sequences 1 \
    --max_episodes 1 \
    --no-wandb \
  2>&1 | tee "$LOG_DIR/step72_rerun.log"

echo ""
echo "============================================================"
echo " 7.2-C  Verifying success criteria -> step72c_verify.log"
echo "============================================================"

{
  echo "===== 7.2-C Check 1: No errors / tracebacks / mismatches ====="
  grep -i "error\|traceback\|mismatch" "$LOG_DIR/step72_rerun.log" \
    && echo "STATUS: ERRORS — check log" \
    || echo "STATUS: [OK] No load errors"

  echo ""
  echo "===== 7.2-C Check 2: Eval loop ran (Evaluating: lines) ====="
  grep "Evaluating:" "$LOG_DIR/step72_rerun.log" \
    && echo "STATUS: [OK] Evaluating lines found" \
    || echo "STATUS: MISSING — no Evaluating: lines"

  echo ""
  echo "===== 7.2-C Check 3: Success rates printed ====="
  grep "success\|grasp\|obj_grasped" "$LOG_DIR/step72_rerun.log" | head -5 \
    && echo "STATUS: [OK] Success rate lines found" \
    || echo "STATUS: MISSING — no success rate lines"

  echo ""
  echo "===== 7.2-C Check 4: Last 5 lines of eval log ====="
  tail -5 "$LOG_DIR/step72_rerun.log"
} 2>&1 | tee "$LOG_DIR/step72c_verify.log"

echo ""
echo "Done. Results are in:"
echo "  $LOG_DIR/step72_rerun.log"
echo "  $LOG_DIR/step72c_verify.log"
