#!/usr/bin/env bash

srun --job-name=eval$1 --gres=gpu:1 python3 scripts/simulation/libero_experiments.py \
    --checkpoint "checkpoints/libero_$1_br_v8" \
    --task_name "libero_$1" \
    --action_len 8
    
sleep 60