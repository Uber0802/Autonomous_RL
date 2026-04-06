#!/bin/bash
# =============================================================================
# Production training for Emu3-based UniVLA with FAST action tokenization.
#
# Uses Yuqi1997/UniVLA/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K + fast-bridge-t5-s50
# (Bridge-fit FAST variant — vocab=1024, scale=50, min_token=-112).
#
# Pipeline:
#   - Vision: Emu3VisionVQ at 384² target (~2160 tokens) — middle ground
#     between 256² (worst quality) and 512² (pretraining grid). Safe choice
#     for 102 GB GPU; bump to 262144 (512²) if you want max quality and have
#     additional headroom. 256² (65536) is the fallback for tighter cards.
#   - Prompt: Emu3Processor.video_process(mode='VLA')
#   - Actions: variable-length FAST BPE → IDCT → [10,7] → first step
#   - Env interface: step_continuous
#   - Buffer:  act_dim=50 (padded), logprob_dim=1 (scalar logprob)
#
# Memory budget (102 GB GPU):
#   - rollout peak ~35–50 GB at 384² (was ~25–35 GB at 256²)
#   - PPO update peak ~55–70 GB at minibatch=1
#   - leaves ~30 GB headroom for spikes / KV cache / fragmentation
#
# Prerequisites (one-time):
#   1. bash setup.sh                       # downloads everything
#   2. ls checkpoints/fast-bridge-t5-s50/  # confirm vocab_size=1024 in
#                                          # processor_config.json
# =============================================================================

set -e
cd "$(dirname "$0")/SimplerEnv"

cuda="0"  # GPU index — pick a free 100+ GB card

# Defensive runtime settings:
#   - expandable_segments     : reduce CUDA fragmentation for big tensors
#   - max_split_size_mb       : cap individual block size to avoid huge holes
#   - GPU_MEM_DEBUG=0         : silence the [mem] phase logger
#   - XLA_PYTHON_...=false    : prevent JAX from grabbing all GPU memory
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPU_MEM_DEBUG=0
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_VERBOSITY=error

CUDA_VISIBLE_DEVICES=$cuda \
python simpler_env/train_ms3_ppo.py \
  --name="univla-emu3-80-seed0" \
  --log="univla-emu3-80-seed0.txt" \
  --wandb_dir=".." \
  --env_id="TwoObjectTwoReceptacle-v1" \
  --vla_type="univla" \
  --vla_path="../checkpoints/univla-emu3-raw/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K" \
  --vision_vq_path="../checkpoints/emu3-vision-tokenizer" \
  --vla_unnorm_key="bridge_robot" \
  --vla_image_pixels=147456 \
  --seed=0 \
  --num_envs=64 \
  --buffer_inferbatch=2 \
  --buffer_minibatch=1 \
  --alg_gradient_accum=8 \
  --training_len=80 \
  --training_interval=80 \
  --max_reset=16384 \
  --max_episodes=256 \
  --interval_eval=16 \
  --interval_save=32
