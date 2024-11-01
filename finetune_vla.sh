#!/usr/bin/env bash

torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/finetune.py \
    --model_path "remyxai/SpaceLLaVA-lite" \
    --action_dim 7 \
    --action_len 1 \
    --learning_rate 1e-5 \
    --hidden_projection "mean" \
    --temp_dir "v1_tmp" \
    --output_dir "v1" \