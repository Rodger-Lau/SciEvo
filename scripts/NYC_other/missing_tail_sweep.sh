#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/NYC

exp_name="missing_tail_1"
python -u main_ST.py \
  --dataset NYC \
  --num_nodes 206 \
  --tod_size 48 \
  --seed 2025 \
  --experiment_name "${exp_name}" \
  --missing_tail_steps 1 \
  > "outputs/BrainAI/NYC/${exp_name}.txt" 2>&1

# exp_name="missing_tail_7"
# python -u main_ST.py \
#   --dataset NYC \
#   --num_nodes 206 \
#   --tod_size 48 \
#   --seed 2025 \
#   --experiment_name "${exp_name}" \
#   --missing_tail_steps 7 \
#   > "outputs/BrainAI/NYC/${exp_name}.txt" 2>&1

# exp_name="missing_tail_7_nodes_25pct"
# python -u main_ST.py \
#   --dataset NYC \
#   --num_nodes 206 \
#   --tod_size 48 \
#   --seed 2025 \
#   --experiment_name "${exp_name}" \
#   --missing_tail_steps 7 \
#   --missing_tail_node_ratio 0.25 \
#   --missing_tail_seed 2025 \
#   > "outputs/BrainAI/NYC/${exp_name}.txt" 2>&1
