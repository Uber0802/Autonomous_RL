WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

DATAPATH=''
ACTION_TOKENIZER_PATH=""
EXP_NAME=""
PRETRAIN=""

export PYTHONPATH=$(pwd)

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path ${PRETRAIN} \
    --model_config_path /share/project/yuqi.wang/UniVLA/configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero3_offload.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 5e-5 \
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
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 100 \
    --per_device_train_batch_size 8 \
    --frames 1 \
    --action_frames 20 \
    --max_position_embeddings 1100 \
    --seed 42 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy no \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_action True \
    --actions True \
    --actions_format "fast" \
    --real_robot True \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
    # --report_to wandb tensorboard \


