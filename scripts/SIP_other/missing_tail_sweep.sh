#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/SIP

exp_name="missing_tail_1"
python -u main_ST.py \
  --dataset SIP \
  --num_nodes 108 \
  --tod_size 288 \
  --seed 2025 \
  --experiment_name "${exp_name}" \
  --missing_tail_steps 1 \
  > "outputs/BrainAI/SIP/${exp_name}.txt" 2>&1

# exp_name="missing_tail_7"
# python -u main_ST.py \
#   --dataset SIP \
#   --num_nodes 108 \
#   --tod_size 288 \
#   --seed 2025 \
#   --experiment_name "${exp_name}" \
#   --missing_tail_steps 7 \
#   > "outputs/BrainAI/SIP/${exp_name}.txt" 2>&1

# exp_name="missing_tail_7_nodes_25pct"
# python -u main_ST.py \
#   --dataset SIP \
#   --num_nodes 108 \
#   --tod_size 288 \
#   --seed 2025 \
#   --experiment_name "${exp_name}" \
#   --missing_tail_steps 7 \
#   --missing_tail_node_ratio 0.25 \
#   --missing_tail_seed 2025 \
#   > "outputs/BrainAI/SIP/${exp_name}.txt" 2>&1
