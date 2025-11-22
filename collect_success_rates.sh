paths=(
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251117_121854-pm6sq4zh
/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251117_124808-debzsrkc
)

output_file="success_summary.txt"

# Run the Python script with all paths as arguments
python3 collect_success_rates.py "${paths[@]}" | tee "$output_file"

echo -e "\n✅ Results saved to: $output_file"