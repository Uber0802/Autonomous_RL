cd SimplerEnv
cuda="0"

vla_load_paths=(
/workspace/Autonomous_RL/SimplerEnv/wandb/run-20251115_122416-mjpckahe/glob/steps_0007
/workspace/Autonomous_RL/SimplerEnv/wandb/run-20251113_230346-j29mm2rw/glob/steps_0001
/workspace/Autonomous_RL/SimplerEnv/wandb/run-20251111_235852-qu0haz91/glob/steps_0003
/workspace/Autonomous_RL/SimplerEnv/wandb/run-20251110_115448-1myi8eex/glob/steps_0003
)

for vla_load_path in "${vla_load_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python simpler_env/train_ms3_ppo.py \
      --env_id="TwoObjectTwoReceptacle-v1" \
      --vla_path="openvla/openvla-7b" --vla_unnorm_key="bridge_orig" \
      --vla_load_path="${vla_load_path}" \
      --seed=2 --no_wandb --only_render_seq
done