#!/bin/bash
# p6_perep_320_evi1.sh — exact launch command for the P-6 v2 PPO run.
#
# Run dir: wandb/run-20260613_042117-6daebzx1  (name: P6_perep_320_evi1)
# Result : PASS (paired McNemar χ²=11.17, p=4.15e-4; in_domain 10.4% → 30.2%;
#          OOD 5.2% → 38.5%, χ²=24.03; all 4 tasks improved 10–33 pp).
# Tagged : phaseP-6  (Autonomous_RL branch spatialvla-integration)
#
# Phase: P-6 (integration done-criterion). The plan called for
# reset_mode=none + reset_unsuitable=True (L9 non-episodic); user revised
# 2026-06-13 to reset_mode=per_episode + horizon 320 after a prior
# reset_mode=none attempt reached 24/32 ep in 28 h. eval_interval=1 to
# match P-5's 128 mid-training eval points env-step-for-env-step.
#
# Env-step budget: 32 × 320 × 64 = 655,360 = P-5's 128 × 80 × 64 (matched).
# Wall-clock: ~48.5 h on RTX PRO 6000 Blackwell (GPU 0).
# Fresh from sft-bridge base (NOT warm-started from the P-5 ckpt).

set -e
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 safejob conda run -n spatialvla_cronos --no-capture-output python main.py \
    --config-path configs/spatialvla_2x2_train.yaml \
    --seed 0 \
    --reset-mode per_episode \
    --episode-len 320 \
    --ppo-update-len 160 \
    --task-len 80 \
    --segment-len 80 \
    --max-episodes 32 \
    --eval-interval 1 \
    --name P6_perep_320_evi1
