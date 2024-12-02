#!/usr/bin/env bash

srun --job-name=eval$1 --gres=gpu:1 python3 scripts/simulation/libero_experiments.py \
    --checkpoint "checkpoints/libero_object_DiT_512_256_DDIM_2gpu" \
    --task_name "libero_$1" \
    --action_len 4
    
sleep 60