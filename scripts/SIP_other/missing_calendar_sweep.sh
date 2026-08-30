#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/BrainAI/SIP

patterns=(daily_1h weekly_1day monthly_1week)

for pattern in "${patterns[@]}"; do
  exp_name="missing_calendar_${pattern}"
  python -u main_ST.py \
    --dataset SIP \
    --num_nodes 108 \
    --tod_size 288 \
    --seed 2025 \
    --experiment_name "$exp_name" \
    --missing_calendar_pattern "$pattern" \
    --missing_calendar_mode random \
    > "outputs/BrainAI/SIP/${exp_name}.txt" 2>&1
done
