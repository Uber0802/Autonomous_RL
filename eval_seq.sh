cd SimplerEnv
cuda="0"

vla_load_paths=(
/path/to/your/dir/SimplerEnv/wandb/run-20260101_123456-abcdefgh/glob/steps_0000
)

for vla_load_path in "${vla_load_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python simpler_env/train_ms3_ppo.py \
      --env_id="TwoObjectTwoReceptacle-v1" \
      --vla_path="openvla/openvla-7b" --vla_unnorm_key="bridge_orig" \
      --vla_load_path="${vla_load_path}" \
      --seed=0 --obj_set="rand" \
      --no_wandb --only_render_seq
done
