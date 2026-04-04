WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

DATAPATH='/share/project/yuqi.wang/datasets/processed_data/meta/libero_all_norm_aug.pkl'
ACTION_TOKENIZER_PATH="/share/project/yuqi.wang/UniVLA/pretrain/fast"
EXP_NAME="UNIVLA_LIBERO_VIDEO_BS192_8k"

export PYTHONPATH=$(pwd)

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path /share/project/yuqi.wang/UniVLA/logs/ckpts/WORLD_MODEL_POSTTRAIN\
    --model_config_path /share/project/yuqi.wang/UniVLA/configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero3_offload.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 8e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 8000 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 50 \
    --per_device_train_batch_size 3 \
    --frames 2 \
    --action_frames 10 \
    --max_position_embeddings 3200 \
    --seed 42 \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 8 \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action True \
    --actions True \
    --actions_format "fast" \
    --use_gripper True \
    --video_format "interleave" \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
