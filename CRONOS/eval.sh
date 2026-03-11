#!/bin/bash
# eval.sh - Run CRONOS evaluation

python main.py \
    --name "CRONOS_Eval" \
    --only_render True \
    --vla_load_path "path/to/checkpoint"
