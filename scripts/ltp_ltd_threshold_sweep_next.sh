#!/usr/bin/env bash
set -euo pipefail

dataset="${DATASET:-NYC}"
seed="${SEED:-2025}"
target_source_idx="${TARGET_SOURCE_IDX:-0}"
target_task_idx="${TARGET_TASK_IDX:-0}"
task_per_dir="${TASK_PER_DIR:-6}"
max_parallel="${MAX_PARALLEL:-2}"
extra_args=("$@")

case "$dataset" in
  CHI)
    num_nodes=220
    tod_size=48
    target_name="RISK"
    ;;
  NYC)
    num_nodes=206
    tod_size=48
    target_name="CROWDIN"
    ;;
  SIP)
    num_nodes=108
    tod_size=288
    target_name="FLOW"
    ;;
  *)
    echo "Unsupported DATASET=$dataset. Use CHI, NYC, or SIP." >&2
    exit 1
    ;;
esac

mkdir -p "outputs/BrainAI/${dataset}"

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$max_parallel" ]]; do
    sleep 30
  done
}

format_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf "%s" "$value"
}

# ltp_ltd_small_thresholds=(
#   0.05
#   0.1
#   0.15
#   0.2
# )
ltp_ltd_small_thresholds=(
  0.1
  0.2
)

for small_threshold in "${ltp_ltd_small_thresholds[@]}"; do
  experiment_name="tune_ltp_ltd_small_$(format_tag "$small_threshold")_${target_name}_task${target_task_idx}"

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
    --save_loss_curves \
    --log_cuda_memory \
    "${extra_args[@]}" \
    --ltp_ltd_small_threshold "$small_threshold" \
    > "outputs/BrainAI/${dataset}/${experiment_name}.txt" 2>&1 &
done

wait
