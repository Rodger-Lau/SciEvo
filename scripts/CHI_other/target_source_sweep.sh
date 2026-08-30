#!/usr/bin/env bash
set -euo pipefail

dataset="CHI"
num_nodes=220
tod_size=48
targets=(0 1 2)
names=(RISK TAXIPICK TAXIDROP)

mkdir -p "outputs/BrainAI/${dataset}"

for idx_pos in "${!targets[@]}"; do
  target_idx="${targets[$idx_pos]}"
  target_name="${names[$idx_pos]}"
  exp_name="target_source_${target_idx}_${target_name}"

  python -u main_ST.py \
    --dataset "$dataset" \
    --num_nodes "$num_nodes" \
    --tod_size "$tod_size" \
    --seed 2025 \
    --target_source_idx "$target_idx" \
    --target_task_idx 0 \
    --experiment_name "$exp_name" \
    "$@" \
    > "outputs/BrainAI/${dataset}/${exp_name}.txt" 2>&1
done
