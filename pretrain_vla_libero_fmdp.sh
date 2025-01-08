#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done
# srun --gres=gpu:$1 
export OMP_NUM_THREADS=64

srun --job-name=lib_$2 --gres=gpu:$1 torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/pretrain.py \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --lora_enable false \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --use_rslora false \
    --weight_decay 1e-6 \
    --data_root_dir "/home/shared/rlds_datasets" \
    --data_mix "libero_$2_no_noops" \
    --output_dir "checkpoints/libero_$2_fmdp" \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1 \
    --adam_epsilon 1e-8 \
    --action_head "FlowMatchingDiffusionPolicy" \
    --action_dim 7 \
    --action_len 8 \
    --max_steps 50000 \
    --save_steps 5000 \
    --shuffle_buffer_size 20000 \
    --batch_size 32 \
    --image_aug false \
    --wandb_project "VLA_LIBERO_DP" \
    --enable_autotune true