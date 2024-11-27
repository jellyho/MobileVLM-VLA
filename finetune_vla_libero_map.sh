#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done
# srun --gres=gpu:$1 
srun --gres=gpu:$1 torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/finetune.py \
    --learning_rate 5e-4 \
    --lr_schedule "constant" \
    --warmup_ratio 0.05 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.01 \
    --use_rslora false \
    --weight_decay 1e-6 \
    --data_root_dir "/home/shared/rlds_datasets" \
    --data_mix "lg_upright_cup" \
    --output_dir "checkpoints/upright_dp_64_rs" \
    --gradient_clip 1.0 \
    --gradient_accumulation_steps 1 \
    --adamw_eps 1e-8 \
    --action_head "DiffusionPolicy" \
    --action_dim 7 \
    --action_len 8 \
    --max_steps 50000 \
    --save_steps 500 \
    --shuffle_buffer_size 10000 \
    --batch_size 32 \
    --image_aug false \
    --wandb_project "VLA_UPRIGHT"