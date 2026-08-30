#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
max_parallel="${MAX_PARALLEL:-5}"
dry_run="${DRY_RUN:-0}"

if ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got $max_parallel." >&2
  exit 2
fi

read -r -a datasets <<< "${DATASETS:-NYC CHI SIP}"
# Two component ablations for each of ProgCA, EvoPoG, and CoML.
modes=(gradc dac datase reinedit mgo arcon)

if [[ "$dry_run" == "1" ]]; then
  for mode in "${modes[@]}"; do
    for dataset in "${datasets[@]}"; do
      DATASET="$dataset" "${script_dir}/_run_ablation.sh" "$mode" "$@"
    done
  done
  exit 0
fi

active_pids=()
declare -A job_labels=()

remove_active_pid() {
  local target_pid="$1"
  local remaining=()
  local pid

  for pid in "${active_pids[@]}"; do
    if [[ "$pid" != "$target_pid" ]]; then
      remaining+=("$pid")
    fi
  done
  active_pids=("${remaining[@]}")
}

terminate_active_runs() {
  local pid

  for pid in "${active_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping ${job_labels[$pid]:-ablation} (PGID $pid)..." >&2
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${active_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  active_pids=()
}

stop_on_signal() {
  trap - INT TERM HUP
  echo >&2
  terminate_active_runs
  exit 130
}

wait_for_one() {
  local finished_pid=""
  local status=0
  local label

  if wait -n -p finished_pid "${active_pids[@]}"; then
    :
  else
    status=$?
  fi

  if [[ -z "$finished_pid" ]]; then
    echo "Unable to identify the completed ablation (status=$status)." >&2
    terminate_active_runs
    exit "$status"
  fi

  label="${job_labels[$finished_pid]}"
  remove_active_pid "$finished_pid"
  unset "job_labels[$finished_pid]"

  if [[ "$status" -ne 0 ]]; then
    echo "Failed: $label (status=$status)." >&2
    terminate_active_runs
    exit "$status"
  fi
  echo "Completed: $label"
}

trap stop_on_signal INT TERM HUP

echo "Running 6 component ablations x ${#datasets[@]} datasets with MAX_PARALLEL=$max_parallel"

for mode in "${modes[@]}"; do
  for dataset in "${datasets[@]}"; do
    if [[ "${#active_pids[@]}" -ge "$max_parallel" ]]; then
      wait_for_one
    fi

    DATASET="$dataset" setsid "${script_dir}/_run_ablation.sh" "$mode" "$@" &
    pid=$!
    active_pids+=("$pid")
    job_labels["$pid"]="no_${mode}/$dataset"
    echo "Started: ${job_labels[$pid]} (PGID $pid)"
  done
done

while [[ "${#active_pids[@]}" -gt 0 ]]; do
  wait_for_one
done
