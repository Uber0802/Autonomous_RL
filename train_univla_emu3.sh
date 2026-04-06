#!/bin/bash
# Warm-up training for Emu3-based UniVLA with FAST action tokenization.
# Uses Yuqi1997/UniVLA/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K + fast-bridge-t5-s50.
#
# Differences from the OpenVLA/Prismatic path:
#   - Vision: Emu3VisionVQ (frozen, producing discrete VQ tokens)
#   - Prompt: Emu3Processor.video_process(mode='VLA')
#   - Actions: variable-length FAST BPE tokens -> IDCT -> [10,7] -> first step
#   - Env interface: step_continuous (like LAM path)
#   - Buffer: act_dim=50 (padded), logprob_dim=1 (scalar)
#   - Smaller vision grid (256^2 pixel target -> ~972 tokens) for grad-ckpt budget

cd SimplerEnv
cuda="1"  # Select GPU (ensure ~40+ GB free)

# Note: no PYTHONPATH tweaks needed — the Emu3 code is imported via relative
# paths inside UniVLA/models/ and UniVLA/reference/Emu3/ .

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="univla-emu3-warmup-seed0" \
  --log="univla-emu3-warmup-seed0.txt" \
  --wandb_dir=".." \
  --env_id="TwoObjectTwoReceptacle-v1" \
  --vla_type="univla" \
  --vla_path="../checkpoints/univla-emu3-raw/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K" \
  --vision_vq_path="../checkpoints/emu3-vision-tokenizer" \
  --vla_unnorm_key="bridge_robot" \
  --seed=0 \
  --num_envs=2 \
  --max_episodes=1 \
  --episode_len=20 \
  --training_len=20 \
  --instruction_switch_interval=20 \
  --training_interval=20 \
  --buffer_inferbatch=2 \
  --no-wandb
