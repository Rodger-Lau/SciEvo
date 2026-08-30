#!/usr/bin/env bash
set -euo pipefail

# Sweep the scale coefficient p0 of the original exponential dropout schedule.
# Run: bash scripts/dropout_schedule_sweep.sh [extra main_ST.py arguments]
# Example: DATASET=CHI TARGET_TASK_IDX=2 MAX_PARALLEL=4 bash scripts/dropout_schedule_sweep.sh
# Uppercase settings below can be overridden with environment variables.

# ==============================================================================
# 1. Basic experiment configuration
# ==============================================================================
dataset="${DATASET:-NYC}"
seed="${SEED:-2025}"
target_source_idx="${TARGET_SOURCE_IDX:-0}"
target_task_idx="${TARGET_TASK_IDX:-0}"
task_per_dir="${TASK_PER_DIR:-6}"
max_parallel="${MAX_PARALLEL:-2}"

# ==============================================================================
# 2. Dropout strategy and hyperparameter sweep configuration
# ==============================================================================
# Format: "<strategy> <hyperparameter> <value>"
# exp uses dropout_p0; linear/inverse/log use dropout_coeff_idx (0, 1, or 2).
# The final dropout probability is clipped to [0.0, 0.9] in main_ST.py.
# Other valid row examples:
#   "linear dropout_coeff_idx 0"
#   "inverse dropout_coeff_idx 1"
#   "log dropout_coeff_idx 2"
run_specs=(
  "exp dropout_p0 0.001"
  "exp dropout_p0 0.01"
  "exp dropout_p0 0.1"
  "exp dropout_p0 1"
)

output_root="${OUTPUT_ROOT:-outputs/BrainAI}"
extra_args=("$@")

# ==============================================================================
# 3. Dataset-specific derived configuration (normally no edits needed)
# ==============================================================================
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

output_dir="${output_root}/${dataset}"
mkdir -p "$output_dir"
child_pids=()

# ==============================================================================
# 4. Helper functions
# ==============================================================================
wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$max_parallel" ]]; do
    sleep 30
  done
}

terminate_children() {
  local pid

  if ((${#child_pids[@]} == 0)); then
    return
  fi

  printf '\nStopping %d training process(es)...\n' "${#child_pids[@]}" >&2
  trap - INT TERM
  for pid in "${child_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${child_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  child_pids=()
}

handle_signal() {
  local signal_name="$1"
  printf '\nReceived %s; stopping this sweep...\n' "$signal_name" >&2
  terminate_children
  if [[ "$signal_name" == "INT" ]]; then
    exit 130
  fi
  exit 143
}

format_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf "%s" "$value"
}

print_configuration() {
  local spec configured_schedule configured_hyperparameter configured_value
  printf '\n============================================================\n'
  printf 'Dropout strategy sweep configuration\n'
  printf '============================================================\n'
  printf '  dataset                 : %s\n' "$dataset"
  printf '  seed                    : %s\n' "$seed"
  printf '  target_source_idx       : %s\n' "$target_source_idx"
  printf '  target_task_idx         : %s\n' "$target_task_idx"
  printf '  task_per_dir            : %s\n' "$task_per_dir"
  printf '  num_nodes               : %s\n' "$num_nodes"
  printf '  tod_size                : %s\n' "$tod_size"
  printf '  target_name             : %s\n' "$target_name"
  printf '  configured runs         :\n'
  for spec in "${run_specs[@]}"; do
    read -r configured_schedule configured_hyperparameter configured_value <<< "$spec"
    printf '    - strategy=%s, %s=%s\n' "$configured_schedule" "$configured_hyperparameter" "$configured_value"
  done
  printf '  max_parallel            : %s\n' "$max_parallel"
  printf '  output_dir              : %s\n' "$output_dir"
  printf '  save_loss_curves        : enabled\n'
  printf '  log_cuda_memory         : enabled\n'
  if ((${#extra_args[@]} > 0)); then
    printf '  extra main_ST.py args   :'
    printf ' %q' "${extra_args[@]}"
    printf '\n'
  else
    printf '  extra main_ST.py args   : <none>\n'
  fi
  printf '============================================================\n'
}

trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap terminate_children EXIT

# ==============================================================================
# 5. Print configuration and launch sweep jobs
# ==============================================================================
print_configuration

total_runs="${#run_specs[@]}"
run_index=0

for spec in "${run_specs[@]}"; do
  read -r dropout_schedule dropout_hyperparameter dropout_value <<< "$spec"
  ((run_index += 1))

  case "${dropout_schedule}:${dropout_hyperparameter}" in
    exp:dropout_p0)
      dropout_args=(--dropout_p0 "$dropout_value")
      hyperparameter_tag="p0_$(format_tag "$dropout_value")"
      ;;
    linear:dropout_coeff_idx|inverse:dropout_coeff_idx|log:dropout_coeff_idx)
      dropout_args=(--dropout_coeff_idx "$dropout_value")
      hyperparameter_tag="c${dropout_value}"
      ;;
    *)
      echo "Unsupported dropout run spec: $spec" >&2
      exit 1
      ;;
  esac

  experiment_name="dropout_${dropout_schedule}_${hyperparameter_tag}_${target_name}_task${target_task_idx}"
  log_file="${output_dir}/${experiment_name}.txt"

  wait_for_slot

  printf '\n[%d/%d] Starting experiment: %s\n' "$run_index" "$total_runs" "$experiment_name"
  printf '      strategy=%s, %s=%s\n' "$dropout_schedule" "$dropout_hyperparameter" "$dropout_value"
  printf '      log_file=%s\n' "$log_file"

  setsid python -u main_ST.py \
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
    --dropout_schedule "$dropout_schedule" \
    "${dropout_args[@]}" \
    > "$log_file" 2>&1 &
  child_pids+=("$!")
done

wait
child_pids=()
trap - INT TERM EXIT
printf '\nAll %d dropout sweep experiments finished.\n' "$total_runs"
