#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done

# srun --gres=gpu:$1 
srun --job-name=DiTnocach --gres=gpu:$1 torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/finetune.py \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --use_rslora false \
    --weight_decay 1e-6 \
    --data_root_dir "/home/shared/vla_benchmark_rlds" \
    --data_mix "bm_pick_tape_single" \
    --output_dir "checkpoints/pick_tape_single_DiT_64_32_DDIM_test" \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1 \
    --adam_epsilon 1e-8 \
    --action_head "DiT" \
    --action_dim 7 \
    --action_len 8 \
    --max_steps 50000 \
    --save_steps 1000 \
    --shuffle_buffer_size 10000 \
    --batch_size 32 \
    --image_aug false \
    --wandb_project "VLA_LIBERO_DiT"