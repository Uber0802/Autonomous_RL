#!/bin/bash
# p5_fresh_128ep.sh — exact launch command for the P-5 128-ep PPO run.
#
# Run dir: wandb/run-20260609_135512-jky97lrf  (name: P5_fresh_128ep)
# Result : PASS (paired McNemar χ²=62.02, p≈0; in_domain 10.4% → 77.1%)
# Tagged : phaseP-5  (Autonomous_RL branch spatialvla-integration)
#
# Phase: P-5 (episodic regime), the easier-regime confirmation gate of
# plans/2026-06-05_spatialvla-ppo-implementation.md. Fresh from the
# sft-bridge base (NOT warm-started). Train config path drives policy=
# spatialvla, vla_path=IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge,
# vla_unnorm_key=bridge_orig/1.0.0, num_envs=64, lr=1e-4/3e-3, etc.
#
# Wall-clock: ~32 h on RTX PRO 6000 Blackwell (GPU 0).

set -e
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 safejob conda run -n spatialvla_cronos --no-capture-output python main.py \
    --config-path configs/spatialvla_2x2_train.yaml \
    --seed 0 \
    --reset-mode per_episode \
    --episode-len 80 \
    --ppo-update-len 80 \
    --task-len 80 \
    --segment-len 80 \
    --max-episodes 128 \
    --name P5_fresh_128ep
