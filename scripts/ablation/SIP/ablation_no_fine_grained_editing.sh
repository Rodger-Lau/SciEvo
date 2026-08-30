#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/SIP
python -u main_ST.py \
  --dataset SIP \
  --num_nodes 108 \
  --tod_size 288 \
  --seed 2025 \
  --experiment_name no_fine_grained_editing \
  --ablation_modules fine_grained_editing \
  > outputs/BrainAI/SIP/no_fine_grained_editing.txt 2>&1