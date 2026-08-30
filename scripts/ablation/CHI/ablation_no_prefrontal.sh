#!/usr/bin/env bash
set -euo pipefail
# CHI NYC SIP
mkdir -p outputs/BrainAI/CHI
python -u main_ST.py \
  --dataset CHI \
  --num_nodes 220 \
  --tod_size 48 \
  --seed 2025 \
  --experiment_name no_prefrontal_decision_making \
  --ablation_modules prefrontal_decision_making \
  > outputs/BrainAI/CHI/no_prefrontal_decision_making.txt 2>&1