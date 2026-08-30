#!/usr/bin/env bash
set -euo pipefail

dataset="NYC"
num_nodes=206
tod_size=48
seed=2025
target_source_idx=0
target_task_idx=0
exp_name="random_baseline"

mkdir -p "outputs/BrainAI/${dataset}"

python -u main_ST.py \
  --dataset "$dataset" \
  --num_nodes "$num_nodes" \
  --tod_size "$tod_size" \
  --seed "$seed" \
  --target_source_idx "$target_source_idx" \
  --target_task_idx "$target_task_idx" \
  --training_strategy random_baseline \
  --experiment_name "$exp_name" \
  "$@" \
  > "outputs/BrainAI/${dataset}/${exp_name}.txt" 2>&1
