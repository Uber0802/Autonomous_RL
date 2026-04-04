#!/bin/bash

# Run
export MESA_GL_VERSION_OVERRIDE=4.1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN          # 启用 NCCL 调试信息
export NCCL_P2P_DISABLE=1        # 禁用 P2P 直连通信
export NCCL_IB_DISABLE=1         # 关闭 InfiniBand
export NCCL_SHM_DISABLE=1        # 禁用共享内存
export NCCL_LAUNCH_MODE=PARALLEL # 避免 NCCL 死锁

emu_hub=$1
GPUS_PER_NODE=8

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=6067 eval/calvin/evaluate_ddp-emu.py \
--emu_hub $emu_hub \
--raw_calvin \
# --debug \
