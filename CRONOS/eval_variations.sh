#!/bin/bash
# eval_variations.sh - Sweep 5 object/receptacle combos on PickPlaceNxM-v1 (N=1, M=1).
#   Original (non-fine-tuned) OpenVLA from HF on all 5 combos.
#   UniVLA (base HF) on the 3 new combos.
#
# obj indices:   1=carrot, 2=kitchen_shovel, 7=ketchup_bottle
# plate indices: 1=yellow_plate, 2=cloth
set -e

ENV_COMMON='--env_id PickPlaceNxM-v1 --env_n 1 --env_m 1 --num_envs 16 --eval_sequential --eval_sequences 5 --record_video --no-wandb --seed 0'
UNIVLA_COMMON='--vla_type univla --vla_path /mnt/home/guest/Cronos_UniVLA/UniVLA/qwbu__univla-7b-224-sft-simpler-bridge --vla_unnorm_key bridge_oxe --univla_window_size 10'
PREFIX='CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PYTHONPATH:$(pwd)'

# Combos: label|obj_idx|plate_idx
COMBOS=(
    "carrot_yellowplate|1|1"
    "shovel_cloth|2|2"
    "ketchup_cloth|7|2"
    "ketchup_yellowplate|7|1"
    "shovel_yellowplate|2|1"
)

# --- Original OpenVLA on all 5 combos ---
for entry in "${COMBOS[@]}"; do
    IFS='|' read -r label obj plate <<< "$entry"
    echo "=============================================="
    echo "OpenVLA (HF base) :: $label (obj=$obj, plate=$plate)"
    echo "=============================================="
    eval $PREFIX python main.py \
        --name "CRONOS_Eval_1x1_openvla_base_${label}" \
        $ENV_COMMON \
        --obj1_index $obj --plate1_index $plate
done

# --- UniVLA on 3 new combos ---
NEW_COMBOS=(
    "ketchup_cloth|7|2"
    "ketchup_yellowplate|7|1"
    "shovel_yellowplate|2|1"
)
for entry in "${NEW_COMBOS[@]}"; do
    IFS='|' read -r label obj plate <<< "$entry"
    echo "=============================================="
    echo "UniVLA :: $label (obj=$obj, plate=$plate)"
    echo "=============================================="
    eval $PREFIX python main.py \
        --name "CRONOS_Eval_1x1_univla_${label}" \
        $ENV_COMMON \
        --obj1_index $obj --plate1_index $plate \
        $UNIVLA_COMMON
done
