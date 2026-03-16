#!/bin/bash
# train.sh - Run CRONOS training

export CUDA_VISIBLE_DEVICES=0,1

# Final Robust Vulkan Configuration
for path in "/usr/share/vulkan/icd.d/nvidia_icd.json" "/etc/vulkan/icd.d/nvidia_icd.json" "/usr/lib/x86_64-linux-gnu/nvidia/vulkan/icd.d/nvidia_icd.json"; do
    if [ -f "$path" ]; then
        export VK_ICD_FILENAMES="$path"
        break
    fi
done
NVIDIA_LIB_PATH=$(find /usr/lib -name "libnvidia-glcore.so*" -print -quit 2>/dev/null | xargs dirname)
if [ -n "$NVIDIA_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIB_PATH:$LD_LIBRARY_PATH"
fi

python main.py \
    --name "CRONOS_Baseline" \
    --env_id "TwoObjectOneReceptacle-v1" \
    --num_envs 64 \
    --seed 0 \
    --wandb True
