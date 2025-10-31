cd SimplerEnv
cuda="0"

vla_load_paths=(
/workspace/Autonomous_RL/SimplerEnv/wandb/run-20251025_153903-40g9323u/glob/steps_0007
)

for vla_load_path in "${vla_load_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python simpler_env/train_ms3_ppo.py \
      --env_id="TwoObjectTwoReceptacle-v1" \
      --vla_path="openvla/openvla-7b" \
      --vla_unnorm_key="bridge_orig" \
      --vla_load_path="${vla_load_path}" \
      --seed=2 --no_wandb --only_render
done