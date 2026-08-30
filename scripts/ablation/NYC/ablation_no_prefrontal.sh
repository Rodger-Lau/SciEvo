#!/usr/bin/env bash
set -euo pipefail
# NYC NYC SIP
mkdir -p outputs/BrainAI/NYC
python -u main_ST.py \
  --dataset NYC \
  --num_nodes 206 \
  --tod_size 48 \
  --seed 2025 \
  --experiment_name no_prefrontal_decision_making \
  --ablation_modules prefrontal_decision_making \
  > outputs/BrainAI/NYC/no_prefrontal_decision_making.txt 2>&1