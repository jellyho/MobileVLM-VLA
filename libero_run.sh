#!/usr/bin/env bash

srun --gres=gpu:1 python3 scripts/simulation/libero_experiments.py \
    --checkpoint "checkpoints/libero_object_dp_alpha32" \
    --task_name "libero_object" \
    --action_len 4