#!/usr/bin/env bash
set -euo pipefail
# SIP SIP SIP
mkdir -p outputs/BrainAI/SIP
python -u main_ST.py \
  --dataset SIP \
  --num_nodes 108 \
  --tod_size 288 \
  --seed 2025 \
  --experiment_name no_prefrontal_decision_making \
  --ablation_modules prefrontal_decision_making \
  > outputs/BrainAI/SIP/no_prefrontal_decision_making.txt 2>&1