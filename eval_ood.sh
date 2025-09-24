cd SimplerEnv
cuda="0"

for vla_load_path in \
    "../SimplerEnv/wandb/run-20250919_122758-zfemajy9/glob/steps_0049" \
    "../SimplerEnv/wandb/run-20250920_222738-qfjqr08d/glob/steps_0031" \
    "../SimplerEnv/wandb/run-20250922_114116-y79iouev/glob/steps_0031"
do
    CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python simpler_env/train_ms3_ppo.py \
      --env_id="TwoObjectTwoReceptacle-v1" \
      --vla_path="openvla/openvla-7b" --vla_unnorm_key="bridge_orig" \
      --vla_load_path="${vla_load_path}" \
      --seed=2 --no_wandb --only_render
done