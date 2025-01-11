#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done
export OMP_NUM_THREADS=64
# srun --gres=gpu:$1 

source  ~/.bashrc

conda activate mobilevlm

cd ~/Bimanual_Imitation/MobileVLM-VLA/

torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node 4 scripts/pretrain.py \
    --learning_rate 2e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.02 \
    --lora_enable false \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --use_rslora false \
    --weight_decay 1e-6 \
    --data_root_dir "/scratch2/jellyho/vla_benchmark_rlds" \
    --data_mix "vla_benchmark" \
    --output_dir "checkpoints/vla_benchmark_octo_4gpu_v5" \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1 \
    --adam_epsilon 1e-8 \
    --action_head "Diffusion" \
    --action_dim 7 \
    --action_len 8 \
    --max_steps 50000 \
    --save_steps 5000 \
    --shuffle_buffer_size 50000 \
    --batch_size 80 \
    --image_aug true \
    --wandb_project "VLA_BENCHMARK_DP"