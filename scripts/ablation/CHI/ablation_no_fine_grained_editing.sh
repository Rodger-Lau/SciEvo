#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/CHI
python -u main_ST.py \
  --dataset CHI \
  --num_nodes 220 \
  --tod_size 48 \
  --seed 2025 \
  --experiment_name no_fine_grained_editing \
  --ablation_modules fine_grained_editing \
  > outputs/BrainAI/CHI/no_fine_grained_editing.txt 2>&1