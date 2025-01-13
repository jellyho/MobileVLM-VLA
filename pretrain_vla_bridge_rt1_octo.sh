#!/usr/bin/env bash

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done
# srun --gres=gpu:$1 
export OMP_NUM_THREADS=16
# export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

srun --job-name=br_rt1 --cpus-per-task=4 --gres=gpu:$1 torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 scripts/pretrain.py \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.01 \
    --lora_enable false \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --use_rslora false \
    --weight_decay 1e-6 \
    --data_root_dir "/data1/OXE" \
    --data_mix "bridge_rt_1" \
    --output_dir "checkpoints/rt1_512_16_4gpu" \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1 \
    --adam_epsilon 1e-8 \
    --action_head "Diffusion" \
    --action_dim 7 \
    --action_len 8 \
    --use_state_input false \
    --state_dim 8 \
    --max_steps 1000000 \
    --save_steps 5000 \
    --shuffle_buffer_size 50000 \
    --batch_size 32 \
    --image_aug true \
    --wandb_project "VLA_BRIDGE_RT_1" \
    --enable_autotune true \
    --resume true