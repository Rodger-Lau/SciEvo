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

# memory_coeffs=(
#   0.0005
#   0.001
#   0.002
#   0.005
# )
memory_coeffs=(
  0.0005
  0.001
)

for memory_coeff in "${memory_coeffs[@]}"; do
  experiment_name="tune_memory_coeff_$(format_tag "$memory_coeff")_${target_name}_task${target_task_idx}"

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
    --memory_coeff "$memory_coeff" \
    > "outputs/BrainAI/${dataset}/${experiment_name}.txt" 2>&1 &
done

wait
