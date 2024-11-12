#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 10000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done

srun --gres=gpu:$1 torchrun --standalone --rdzv_id=$SLURM_JOB_ID --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/finetune.py \
    --learning_rate 8e-5 \
    --lora_rank 64 \
    --lora_alpha 32 \
    --use_rslora true \
    --weight_decay 0.1 \
    --data_mix "lg_cup_color_rightarm" \
    --output_dir "checkpoints/lora-64-32-rs-2gpu" \
    --gradient_clip 1.0 \
    --action_head "MAP" \
    --max_steps 50000 \
    --save_steps 500 \
    --shuffle_buffer_size 10000 \
    --batch_size 32 \
    --image_aug true \
    --wandb_project "vla_cup_color_rightarm"