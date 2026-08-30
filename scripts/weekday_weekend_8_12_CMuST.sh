#!/usr/bin/env bash
set -euo pipefail

dataset="${DATASET:-CHI}"
seed="${SEED:-2025}"
experiment_name="${EXPERIMENT_NAME:-weekday_weekend_8_12_CMuST}"
source_epochs="${SOURCE_EPOCHS:-10}"
target_epochs="${TARGET_EPOCHS:-50}"
batch_size="${BATCH_SIZE:-16}"
target_source_idx="${TARGET_SOURCE_IDX:-0}"

case "$dataset" in
  CHI)
    num_nodes=220
    tod_size=48
    ;;
  *)
    echo "Unsupported DATASET=$dataset. This script is for CHI/RISK by default." >&2
    exit 1
    ;;
esac

mkdir -p "outputs/BrainAI/${dataset}"

python -u main_ST.py \
  --dataset "$dataset" \
  --num_nodes "$num_nodes" \
  --tod_size "$tod_size" \
  --seed "$seed" \
  --experiment_name "$experiment_name" \
  --target_source_idx "$target_source_idx" \
  --task_per_dir 6 \
  --target_task_idx 2 \
  --training_strategy random_baseline \
  --model_name CMuST \
  --batch_size "$batch_size" \
  --baseline_source_epochs "$source_epochs" \
  --baseline_target_epochs "$target_epochs" \
  --report_weekday_weekend_metrics \
  --report_hour_start 8 \
  --report_hour_end 12 \
  "$@" \
  > "outputs/BrainAI/${dataset}/${experiment_name}.txt" 2>&1
