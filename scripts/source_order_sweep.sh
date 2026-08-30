#!/usr/bin/env bash
set -euo pipefail

dataset="${DATASET:-NYC}"
seed="${SEED:-2025}"
target_source_idx="${TARGET_SOURCE_IDX:-0}"
target_task_idx="${TARGET_TASK_IDX:-0}"
task_per_dir="${TASK_PER_DIR:-6}"
report_hour_start="${REPORT_HOUR_START:-8}"
report_hour_end="${REPORT_HOUR_END:-12}"
max_parallel="${MAX_PARALLEL:-1}"

case "$dataset" in
  NYC)
    num_nodes=206
    tod_size=48
    target_name="CROWDIN"
    ;;
  *)
    echo "Unsupported DATASET=$dataset for this sweep. Use NYC." >&2
    exit 1
    ;;
esac

mkdir -p "outputs/BrainAI/${dataset}"

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$max_parallel" ]]; do
    sleep 30
  done
}

source_orders=(
  domain_reverse
  temporal_reverse
  random
  reverse
)

for source_order in "${source_orders[@]}"; do
  experiment_name="source_order_${source_order}_${target_name}_task${target_task_idx}"

  wait_for_slot

  python -u main_ST.py \
    --dataset "$dataset" \
    --num_nodes "$num_nodes" \
    --tod_size "$tod_size" \
    --seed "$seed" \
    --experiment_name "$experiment_name" \
    --target_source_idx "$target_source_idx" \
    --task_per_dir "$task_per_dir" \
    --target_task_idx "$target_task_idx" \
    --source_order "$source_order" \
    --save_loss_curves \
    --report_weekday_weekend_metrics \
    --report_hour_start "$report_hour_start" \
    --report_hour_end "$report_hour_end" \
    "$@" \
    > "outputs/BrainAI/${dataset}/${experiment_name}.txt" 2>&1 &
done

wait
