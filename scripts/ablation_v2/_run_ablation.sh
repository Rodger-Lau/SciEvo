#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODE [extra main_ST.py args...]" >&2
  echo "MODE must be one of: gradc, dac, datase, reinedit, mgo, arcon" >&2
  exit 2
fi

mode="$1"
shift

case "$mode" in
  gradc|dac|datase|reinedit|mgo|arcon) ;;
  *)
    echo "Unsupported MODE='$mode'. Use gradc, dac, datase, reinedit, mgo, or arcon." >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

dataset="${DATASET:-NYC}"
seed="${SEED:-2025}"
gpu="${GPU:-0}"
target_source_idx="${TARGET_SOURCE_IDX:-0}"
target_task_idx="${TARGET_TASK_IDX:-0}"
task_per_dir="${TASK_PER_DIR:-6}"
dry_run="${DRY_RUN:-0}"
python_bin="${PYTHON_BIN:-python}"

case "$dataset" in
  NYC)
    num_nodes=206
    tod_size=48
    source_names=(CROWDIN CROWDOUT TAXIDROP TAXIPICK)
    ;;
  CHI)
    num_nodes=220
    tod_size=48
    source_names=(RISK TAXIPICK TAXIDROP)
    ;;
  SIP)
    num_nodes=108
    tod_size=288
    source_names=(FLOW SPEED)
    ;;
  *)
    echo "Unsupported DATASET='$dataset'. Use NYC, CHI, or SIP." >&2
    exit 2
    ;;
esac

for integer_setting in \
  "SEED:$seed" \
  "GPU:$gpu" \
  "TARGET_SOURCE_IDX:$target_source_idx" \
  "TARGET_TASK_IDX:$target_task_idx" \
  "TASK_PER_DIR:$task_per_dir"; do
  setting_name="${integer_setting%%:*}"
  setting_value="${integer_setting#*:}"
  if ! [[ "$setting_value" =~ ^[0-9]+$ ]]; then
    echo "${setting_name} must be a non-negative integer, got '$setting_value'." >&2
    exit 2
  fi
done

if [[ "$target_source_idx" -ge "${#source_names[@]}" ]]; then
  echo "TARGET_SOURCE_IDX=$target_source_idx is out of range for $dataset (0..$(("${#source_names[@]}" - 1)))." >&2
  exit 2
fi
target_source_name="${source_names[$target_source_idx]}"

if [[ "$task_per_dir" -eq 0 ]]; then
  echo "TASK_PER_DIR must be greater than zero." >&2
  exit 2
fi

if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1, got '$dry_run'." >&2
  exit 2
fi

experiment_name="${EXPERIMENT_NAME:-ablation_v2_no_${mode}_${dataset}_${target_source_name}_task${target_task_idx}_seed${seed}}"
log_dir="${LOG_DIR:-outputs/BrainAI/${dataset}/ablation_v2}"
log_file="${LOG_FILE:-${log_dir}/${experiment_name}.txt}"

cmd=(
  "$python_bin" -u main_ST.py
  --dataset "$dataset"
  --num_nodes "$num_nodes"
  --tod_size "$tod_size"
  --gpu "$gpu"
  --seed "$seed"
  --target_source_idx "$target_source_idx"
  --target_task_idx "$target_task_idx"
  --task_per_dir "$task_per_dir"
  --training_strategy proposed
  --model_name CMuST
  --num_random_runs 1
  --experiment_name "$experiment_name"
  --ablation_modules "$mode"
  "$@"
)

echo "Ablation v2"
echo "  mode          : no_${mode}"
echo "  dataset       : ${dataset}"
echo "  target source : ${target_source_name} (index ${target_source_idx})"
echo "  target task   : ${target_task_idx}"
echo "  task_per_dir  : ${task_per_dir}"
echo "  seed          : ${seed}"
echo "  experiment    : ${experiment_name}"
echo "  log           : ${log_file}"

if [[ "$dry_run" == "1" ]]; then
  printf 'DRY RUN command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$log_dir"
echo "Starting training..."
"${cmd[@]}" >"$log_file" 2>&1
echo "Completed. Log: $log_file"
