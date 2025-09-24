cd SimplerEnv

#cuda="0,1" # env on GPU-0, model on GPU-1 (for 40G GPU)
cuda="0" # env and model on the same GPU (for 80G GPU)

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="bottle_shovel-320-train_once-joint-reset-seed_2" \
  --log="/workspace/Autonomous_RL/bottle_shovel-320-train_once-joint-reset-seed_2.txt" \
  --env_id="TwoObjectTwoReceptacle-v1" \
  --vla_path="openvla/openvla-7b" --vla_unnorm_key="bridge_orig" \
  --seed=2