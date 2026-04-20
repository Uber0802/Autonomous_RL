#!/bin/bash
# Fill missing fine-tuned OpenVLA (seed0) cells for 3 combos.
set -e

ENV_COMMON='--env_id PickPlaceNxM-v1 --env_n 1 --env_m 1 --num_envs 16 --eval_sequential --eval_sequences 5 --record_video --no-wandb --seed 0'
FT_PATH='/mnt/home/ChengDian/workspace/TestCheckpoint/seed0'
PREFIX='CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PYTHONPATH:$(pwd)'

COMBOS=(
    "ketchup_cloth|7|2"
    "ketchup_yellowplate|7|1"
    "shovel_yellowplate|2|1"
)

for entry in "${COMBOS[@]}"; do
    IFS='|' read -r label obj plate <<< "$entry"
    echo "=============================================="
    echo "OpenVLA (fine-tuned seed0) :: $label (obj=$obj, plate=$plate)"
    echo "=============================================="
    eval $PREFIX python main.py \
        --name "CRONOS_Eval_1x1_openvla_ft_${label}" \
        $ENV_COMMON \
        --obj1_index $obj --plate1_index $plate \
        --vla_load_path "$FT_PATH"
done
