#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done

srun --gres=gpu:$1 torchrun --standalone --rdzv_id=$SLURM_JOB_ID --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/finetune.py \
    --learning_rate 1e-4 \
    --lr_schedule "cosine" \
    --lora_rank 64 \
    --lora_alpha 16 \
    --use_rslora false \
    --weight_decay 0.01 \
    --data_root_dir "/home/shared/rlds_datasets" \
    --data_mix "lg_cup_color_rightarm" \
    --output_dir "checkpoints/cup_color_rightarm_dp" \
    --gradient_clip 1.0 \
    --gradient_accumulation_steps 8 \
    --adamw_eps 1e-5 \
    --action_head "Diffusion" \
    --action_dim 7 \
    --action_len 8 \
    --max_steps 50000 \
    --save_steps 500 \
    --shuffle_buffer_size 10000 \
    --batch_size 32 \
    --image_aug false \
    --wandb_project "vla_cup_color_rightarm"