#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/CHI
python -u main_ST.py \
  --dataset CHI \
  --num_nodes 220 \
  --tod_size 48 \
  --seed 2025 \
  --experiment_name no_learning_memory_interaction \
  --ablation_modules learning_memory_interaction \
  > outputs/BrainAI/CHI/no_learning_memory_interaction.txt 2>&1