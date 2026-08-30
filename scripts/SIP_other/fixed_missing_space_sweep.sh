#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/SIP

ratios=(0.1 0.4 0.8)

for ratio in "${ratios[@]}"; do
  pct=$(printf "%.0f" "$(awk "BEGIN {print $ratio * 100}")")
  exp_name="fixed_missing_space_${pct}pct"
  python -u main_ST.py \
    --dataset SIP \
    --num_nodes 108 \
    --tod_size 288 \
    --seed 2025 \
    --experiment_name "$exp_name" \
    --missing_node_ratio "$ratio" \
    > "outputs/BrainAI/SIP/${exp_name}.txt" 2>&1
done
