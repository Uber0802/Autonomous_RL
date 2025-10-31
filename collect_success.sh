SCRIPT="success_collector.py"

paths=(
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_032638-4y0vbvac
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_035347-7vmq8ham
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_042120-6ttcg477
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_044904-ymtjgz4l
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_051643-4nrzhvl5
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_054345-tzt1uskk
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_061139-o2vvvzzp
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_072003-qrhxl5lp
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_072231-6y5vkh3z
)

# Loop through each directory argument
for DIR in "${paths[@]}"; do
  if [ -d "$DIR" ]; then
    echo "📂 Processing $DIR ..."
    python3 "$SCRIPT" "$DIR"
  else
    echo "⚠️ Skipping $DIR (not a directory)"
  fi
done

echo "✅ All done!"