#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done
# srun --gres=gpu:$1 
export OMP_NUM_THREADS=16

srun --gres=gpu:$1 --job-name=vla_bench torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/finetune.py \
    --learning_rate 4e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --lora_rank 128 \
    --lora_alpha  64 \
    --lora_dropout 0.01 \
    --use_rslora false \
    --weight_decay 1e-6 \
    --data_root_dir "/home/shared/vla_benchmark_rlds" \
    --data_mix "vla_benchmark" \
    --output_dir "checkpoints/vla_benchmark_dp_128_64_8_$1gpu" \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1 \
    --adam_epsilon 1e-8 \
    --action_head "DiffusionPolicy" \
    --action_dim 7 \
    --action_len 8 \
    --max_steps 100000 \
    --save_steps 1000 \
    --shuffle_buffer_size 20000 \
    --batch_size 32 \
    --image_aug true \
    --wandb_project "VLA_BENCHMARK_DP"
    
sleep 60