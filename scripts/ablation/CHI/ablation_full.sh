#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/CHI
python -u main_ST.py \
  --dataset CHI \
  --num_nodes 220 \
  --tod_size 48 \
  --seed 2025 \
  --experiment_name full \
  > outputs/BrainAI/CHI/full.txt 2>&1