# cd SimplerEnv

#cuda="0,1" # env on GPU-0, model on GPU-1 (for 40G GPU)
cuda="0" # env and model on the same GPU (for 80G GPU)

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. python SimplerEnv/simpler_env/residual_ppo.py \
  --name="autoRL3_residual_SD_earlyguide" \
  --env_id="TwoObjectTwoReceptacle-v1" \
  --vla_path="openvla/openvla-7b" --vla_unnorm_key="bridge_orig" \
  --seed=2
