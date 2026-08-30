#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/SIP
python -u main_ST.py \
  --dataset SIP \
  --num_nodes 108 \
  --tod_size 288 \
  --seed 2025 \
  --experiment_name no_learning_memory_interaction \
  --ablation_modules learning_memory_interaction \
  > outputs/BrainAI/SIP/no_learning_memory_interaction.txt 2>&1