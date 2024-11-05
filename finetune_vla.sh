#!/usr/bin/env bash

torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/finetune.py \
    --learning_rate 1e-5 \
    --data_mix "lg_stack_cup_5hz" \
    --output_dir "checkpoints/" \
    --max_steps 200000