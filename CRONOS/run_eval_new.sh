#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3
python main.py \
  --num-envs 4 \
  --eval-sequences 2 \
  --segment-len 40 \
  --vla-path openvla/openvla-7b \
  --vla-load-path test_checkpoints/test \
  --name CRONOS-Baseline-Eval-AutoRL-v3 \
  --wandb \
  --obj1-index 7 --obj2-index 2 \
  --plate1-index 1 --plate2-index 2 \
  --eval_sequential
