#!/usr/bin/env bash
set -euo pipefail

seed="${SEED:-2025}"
experiment_name="${EXPERIMENT_NAME:-weekday_weekend_8_12}"

run_experiment() {
  local dataset="$1"
  local target_source_idx="$2"
  shift 2

  local num_nodes
  local tod_size

  case "$dataset" in
    CHI)
      num_nodes=220
      tod_size=48
      ;;
    NYC)
      num_nodes=206
      tod_size=48
      ;;
    *)
      echo "Unsupported DATASET=$dataset. Use CHI or NYC." >&2
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
    --report_weekday_weekend_metrics \
    --report_hour_start 8 \
    --report_hour_end 12 \
    "$@" \
    > "outputs/BrainAI/${dataset}/${experiment_name}.txt" 2>&1
}

run_experiment CHI 0 "$@"
