#!/usr/bin/env bash

python3 scripts/finetune.py \
    --model_path "remyxai/SpaceLLaVA-lite" \
    --action_hidden_size 256 \
    --action_dim 14 \
    --action_len 1 \
