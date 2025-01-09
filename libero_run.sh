#!/usr/bin/env bash

srun --job-name=eval$1 --gres=gpu:1 python3 spatialvla/simulation/libero_experiments.py \
    --checkpoint "checkpoints/libero_$1_dp_v2" \
    --task_name "libero_$1" \
    --action_len 8 \
    --num_trials_per_task 50
    
sleep 60