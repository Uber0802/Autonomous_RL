#!/bin/bash
# eval_2x2.sh - Per-task eval on a 2-object / 2-receptacle scene.
# Scene: ketchup bottle (obj idx 7) + kitchen shovel (obj idx 2);
#        yellow_plate (plate idx 1) + cloth (plate idx 2).
# Task pool (4 tasks): all (object, receptacle) combos.
set -e

ENV_COMMON='--env_id PickPlaceNxM-v1 --env_n 2 --env_m 2 --obj1_index 7 --obj2_index 2 --plate1_index 1 --plate2_index 2 --num_envs 16 --eval_single --record_video --no-wandb --seed 0'
UNIVLA='--vla_type univla --vla_path /mnt/home/guest/Cronos_UniVLA/UniVLA/qwbu__univla-7b-224-sft-simpler-bridge --vla_unnorm_key bridge_oxe --univla_window_size 10'
FT_PATH='/mnt/home/ChengDian/workspace/TestCheckpoint/seed0'
PREFIX='CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PYTHONPATH:$(pwd)'

echo "=============================================="
echo "OpenVLA (HF base) :: 2x2"
echo "=============================================="
eval $PREFIX python main.py \
    --name "CRONOS_Eval_2x2_openvla_base" \
    $ENV_COMMON

echo "=============================================="
echo "OpenVLA (fine-tuned seed0) :: 2x2"
echo "=============================================="
eval $PREFIX python main.py \
    --name "CRONOS_Eval_2x2_openvla_ft" \
    $ENV_COMMON \
    --vla_load_path "$FT_PATH"

echo "=============================================="
echo "UniVLA (HF base) :: 2x2"
echo "=============================================="
eval $PREFIX python main.py \
    --name "CRONOS_Eval_2x2_univla" \
    $ENV_COMMON \
    $UNIVLA
