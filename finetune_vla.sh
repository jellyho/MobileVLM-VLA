#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done

srun --gres=gpu:$1 torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/finetune.py \
    --learning_rate 3e-4 \
    --lr_schedule "linear" \
    --warmup_ratio 0.06 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --use_rslora false \
    --weight_decay 0.1 \
    --data_root_dir "/home/shared/rlds_datasets" \
    --data_mix "libero_object_no_noops" \
    --output_dir "checkpoints/libero_object_dp_small" \
    --gradient_clip 1.0 \
    --gradient_accumulation_steps 1 \
    --adamw_eps 1e-8 \
    --action_head "Diffusion" \
    --action_dim 7 \
    --action_len 5 \
    --max_steps 50000 \
    --save_steps 500 \
    --shuffle_buffer_size 10000 \
    --batch_size 32 \
    --image_aug false \
    --wandb_project "vla_cup_color_rightarm"