#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done
export OMP_NUM_THREADS=16
# srun --gres=gpu:$1 
srun --job-name=twinvla_test --cpus-per-task=16 --gres=gpu:$1 torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/pretrain_twinvla.py \
    --single_model_path 'checkpoints/vla_benchmark_octo_full_1gpu_v2' \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --lora_enable false \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --use_rslora false \
    --weight_decay 1e-6 \
    --data_root_dir "/home/shared/" \
    --data_mix "transfer_cup" \
    --output_dir "checkpoints/twinvla_transfer_cup" \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 4 \
    --adam_epsilon 1e-8 \
    --action_head "Diffusion" \
    --action_dim 7 \
    --action_len 8 \
    --max_steps 50000 \
    --save_steps 5000 \
    --shuffle_buffer_size 10000 \
    --batch_size 8 \
    --image_aug true \
    --wandb_project "VLA_BENCHMARK_DP"