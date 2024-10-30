#!/usr/bin/env bash

torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/finetune.py \
    --model_path "remyxai/SpaceLLaVA-lite" \
    --action_hidden_size 256 \
    --action_dim 7 \
    --action_len 1 \
    --learning_rate 1e-4 \
    --temp_dir "SpatialVLA_v2_layernorm_temp" \
    --output_dir "SpatialVLA_v2_layernorm" \
    --action_layernorm true