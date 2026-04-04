cuda="${cuda:-0}"

cd "$(dirname "$0")/SimplerEnv"

export CUDA_VISIBLE_DEVICES=$cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python simpler_env/bc_eval.py \
        --bc_init_path ../bc_checkpoints/gaussian_head_init/bc_init.pt \
        --env_id TwoObjectTwoReceptacle-v1 \
        --num_episodes 3 \
        --episode_len 80