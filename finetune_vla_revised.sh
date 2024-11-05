#!/usr/bin/env bash

torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/finetune_revised.py \
    --learning_rate 1e-5 \
    --output_dir "checkpoints/v1_test" \
    --max_steps 200000