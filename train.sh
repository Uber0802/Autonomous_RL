cd SimplerEnv
cuda="0" # Select GPU

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="cronos-bottle_shovel-320-rand_scene-seed_0" \
  --log="cronos-bottle_shovel-320-rand_scene-seed_0.txt" \
  --wandb_dir=".." \
  --env_id="TwoObjectTwoReceptacle-v1" \
  --vla_path="openvla/openvla-7b" --vla_unnorm_key="bridge_orig" \
  --seed=0
