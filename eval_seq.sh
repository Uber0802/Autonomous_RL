cd SimplerEnv
cuda="0"

vla_load_paths=(
/workspace/Autonomous_RL/runs/bottle_shovel-320-cont-rand_reset-reset_unsuitable-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-320-cont-rand_reset-rm_unsuite-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-320-cont-rand_reset-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-320-cont1006-rand_reset-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-320-cont1014-rand_resetgrip-train_once-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-320-cont320fs-rand_reset-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-320-rand_reset-BC-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-320-rand_reset-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-320-train_twice-joint-reset-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-640-cont-rand_reset-seed_2
/workspace/Autonomous_RL/runs/bottle_shovel-640-cont1014-rand_resetgrip-train_once-seed_2
)

for vla_load_path in "${vla_load_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python simpler_env/train_ms3_ppo.py \
      --env_id="TwoObjectTwoReceptacle-v1" \
      --vla_path="openvla/openvla-7b" --vla_unnorm_key="bridge_orig" \
      --vla_load_path="${vla_load_path}" \
      --seed=2 --no_wandb --only_render_seq
done