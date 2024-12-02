#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done

# srun --gres=gpu:$1 
export OMP_NUM_THREADS=8
srun --job-name=DiT-LB-P --gres=gpu:$1 torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/finetune.py \
    --learning_rate 1.414e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --lora_rank 512 \
    --lora_alpha 256 \
    --lora_dropout 0.01 \
    --use_rslora false \
    --weight_decay 0.0 \
    --data_root_dir "/home/shared/rlds_datasets" \
    --data_mix "libero_object_no_noops" \
    --output_dir "checkpoints/libero_object_DiT_512_256_DDIM_2gpu" \
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
    --wandb_project "VLA_LIBERO_DiT" \
    --enable_autotune true