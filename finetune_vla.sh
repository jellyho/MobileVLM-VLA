#!/usr/bin/env bash

srun --gres=gpu:$1 torchrun --standalone --nnodes 1 --nproc-per-node $1 scripts/finetune.py \
    --learning_rate 1e-5 \
    --data_mix "lg_stack_cup_5hz" \
    --output_dir "checkpoints/ddp" \
    --max_steps 50000 \
    --save_steps 500 \
    --shuffle_buffer_size 100000 \
    --batch_size 32 \
    --image_aug true