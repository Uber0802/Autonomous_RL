#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export ARNOLD_WORKER_GPU=1
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export RANK=0

export OMP_NUM_THREADS=16

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"

ckpt_dir=$1
GPUS_PER_NODE=$ARNOLD_WORKER_GPU

python eval/libero/evaluate_libero_emu.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_goal \
--cache_root /share/project/yuqi.wang/UniVLA/logs/libero \
# --debug
