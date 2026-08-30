#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/NYC

ratios=(0.1)

for ratio in "${ratios[@]}"; do
  pct=$(printf "%.0f" "$(awk "BEGIN {print $ratio * 100}")")
  exp_name="fixed_missing_space_${pct}pct"
  python -u main_ST.py \
    --dataset NYC \
    --num_nodes 206 \
    --tod_size 48 \
    --seed 2025 \
    --experiment_name "$exp_name" \
    --missing_node_ratio "$ratio" \
    > "outputs/BrainAI/NYC/${exp_name}.txt" 2>&1
done
