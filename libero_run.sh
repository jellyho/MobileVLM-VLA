#!/usr/bin/env bash

srun --gres=gpu:1 python3 scripts/simulation/libero_experiments.py \
    --checkpoint "checkpoints/libero_object_map_bf" \
    --task_name "libero_object"