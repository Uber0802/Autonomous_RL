cd SimplerEnv
cuda="0"

# Run the training script with the specified configuration
CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="test" \
  --env_id="TwoCarrotTwoPlate" \
  --vla_path="openvla/openvla-7b" \
  --vla_unnorm_key="bridge_orig" \
  --seed=2