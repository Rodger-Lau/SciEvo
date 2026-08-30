#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

read -r -a models <<< "${MODELS:-STGODE CMuST STGCN}"

num_runs="${NUM_RUNS:-5}"
base_seed="${BASE_SEED:-2025}"
source_epochs="${SOURCE_EPOCHS:-10}"
target_epochs="${TARGET_EPOCHS:-50}"
batch_size="${BATCH_SIZE:-16}"
gpu="${GPU:-0}"
max_parallel="${MAX_PARALLEL:-2}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

if ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got $max_parallel." >&2
  exit 1
fi

if [[ "$skip_completed" != "0" && "$skip_completed" != "1" ]]; then
  echo "SKIP_COMPLETED must be 0 or 1, got $skip_completed." >&2
  exit 1
fi

if [[ -n "${SEEDS:-}" ]]; then
  read -r -a seeds <<< "$SEEDS"
else
  seeds=()
  for ((run_idx = 0; run_idx < num_runs; run_idx++)); do
    seeds+=("$((base_seed + run_idx))")
  done
fi

if [[ "${#seeds[@]}" -eq 0 ]]; then
  echo "No seeds configured. Set NUM_RUNS or SEEDS." >&2
  exit 1
fi

for model in "${models[@]}"; do
  case "$model" in
    STGODE|CMuST|STGCN) ;;
    *)
      echo "Unsupported model '$model'. Use STGODE, CMuST, or STGCN." >&2
      exit 1
      ;;
  esac
done

dataset="${DATASET:-NYC}"
target_source_idx=0
target_task_idx=0
task_per_dir=6

case "$dataset" in
  NYC)
    num_nodes=206
    tod_size=48
    target_name="CROWDIN"
    ;;
  CHI)
    num_nodes=220
    tod_size=48
    target_name="RISK"
    ;;
  SIP)
    num_nodes=108
    tod_size=288
    target_name="FLOW"
    ;;
  *)
    echo "Unsupported DATASET=$dataset. Use NYC, CHI, or SIP." >&2
    exit 1
    ;;
esac

active_pids=()
declare -A job_labels=()
declare -A job_logs=()

remove_active_pid() {
  local target_pid="$1"
  local remaining_pids=()
  local pid

  for pid in "${active_pids[@]}"; do
    if [[ "$pid" != "$target_pid" ]]; then
      remaining_pids+=("$pid")
    fi
  done
  active_pids=("${remaining_pids[@]}")
}

terminate_active_runs() {
  local pid

  for pid in "${active_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping ${job_labels[$pid]:-experiment} (PGID $pid)..." >&2
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
  local log_file

  if wait -n -p finished_pid "${active_pids[@]}"; then
    :
  else
    status=$?
  fi

  if [[ -z "$finished_pid" ]]; then
    echo "Unable to identify the completed experiment (status=$status)." >&2
    terminate_active_runs
    exit "$status"
  fi

  label="${job_labels[$finished_pid]}"
  log_file="${job_logs[$finished_pid]}"
  remove_active_pid "$finished_pid"
  unset "job_labels[$finished_pid]" "job_logs[$finished_pid]"

  if [[ "$status" -ne 0 ]]; then
    echo "Experiment failed: $label (status=$status)." >&2
    echo "See $log_file" >&2
    terminate_active_runs
    exit "$status"
  fi
  echo "Completed: $label"
}

trap stop_on_signal INT TERM HUP

result_group="baseline_repeats_${dataset}_${target_name}_0_4"
plot_dir="csv_files/BrainAI/${dataset}/${result_group}"
mkdir -p "outputs/BrainAI/${dataset}" "$plot_dir"

echo "$dataset/$target_name/00:00-04:00 repeated baseline experiments"
echo "Models : ${models[*]}"
echo "Seeds  : ${seeds[*]}"
echo "Runs   : ${#seeds[@]} per model"
echo "Parallel jobs: $max_parallel"
echo "Skip completed: $skip_completed"

for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    exp_name="${result_group}_${model}_seed${seed}"
    log_file="outputs/BrainAI/${dataset}/${exp_name}.txt"
    summary_file="csv_files/BrainAI/${dataset}/${exp_name}/summary.csv"

    cmd=(
      python -u main_ST.py
      --dataset "$dataset"
      --num_nodes "$num_nodes"
      --tod_size "$tod_size"
      --gpu "$gpu"
      --seed "$seed"
      --target_source_idx "$target_source_idx"
      --target_task_idx "$target_task_idx"
      --task_per_dir "$task_per_dir"
      --training_strategy random_baseline
      --model_name "$model"
      --experiment_name "$exp_name"
      --batch_size "$batch_size"
      --baseline_source_epochs "$source_epochs"
      --baseline_target_epochs "$target_epochs"
      --num_random_runs 1
      --disable_sample_predictions
    )

    echo
    echo "[$model][seed=$seed] log: $log_file"
    if [[ "$skip_completed" == "1" && -f "$summary_file" ]]; then
      echo "Skipped completed: model=$model seed=$seed ($summary_file)"
      continue
    fi
    if [[ "$dry_run" == "1" ]]; then
      printf '  '
      printf '%q ' "${cmd[@]}"
      printf '\n'
      continue
    fi

    if [[ "${#active_pids[@]}" -ge "$max_parallel" ]]; then
      wait_for_one
    fi

    setsid "${cmd[@]}" > "$log_file" 2>&1 &
    pid=$!
    active_pids+=("$pid")
    job_labels["$pid"]="model=$model seed=$seed"
    job_logs["$pid"]="$log_file"
    echo "Started: model=$model seed=$seed (PGID $pid)"
  done
done

while [[ "${#active_pids[@]}" -gt 0 ]]; do
  wait_for_one
done

if [[ "$dry_run" == "1" ]]; then
  echo
  echo "DRY_RUN=1: experiments and plotting were not executed."
  exit 0
fi

CITY_REPEAT_MODELS="${models[*]}" \
CITY_REPEAT_SEEDS="${seeds[*]}" \
CITY_REPEAT_GROUP="$result_group" \
CITY_REPEAT_DATASET="$dataset" \
CITY_REPEAT_TARGET="$target_name" \
python - "$plot_dir" <<'PY'
import ast
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plot_dir = Path(sys.argv[1])
models = os.environ["CITY_REPEAT_MODELS"].split()
seeds = os.environ["CITY_REPEAT_SEEDS"].split()
group = os.environ["CITY_REPEAT_GROUP"]
dataset = os.environ["CITY_REPEAT_DATASET"]
target = os.environ["CITY_REPEAT_TARGET"]
metrics = ("mae", "rmse", "mape")
rows = []

for model in models:
    for run_number, seed in enumerate(seeds, start=1):
        exp_name = f"{group}_{model}_seed{seed}"
        summary_path = Path("csv_files/BrainAI") / dataset / exp_name / "summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing result: {summary_path}")

        with summary_path.open(newline="") as handle:
            summary = next(csv.DictReader(handle))

        row = {
            "model": model,
            "run": run_number,
            "seed": int(seed),
        }
        for metric in metrics:
            values = ast.literal_eval(summary[f"{metric}_all"])
            if len(values) != 1:
                raise ValueError(
                    f"{summary_path}: expected one value in {metric}_all, got {values}"
                )
            row[metric] = float(values[0])
        rows.append(row)

metrics_path = plot_dir / "repeat_metrics.csv"
with metrics_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=["model", "run", "seed", "mae", "rmse", "mape"]
    )
    writer.writeheader()
    writer.writerows(rows)

colors = {
    "STGODE": "#4C78A8",
    "CMuST": "#F58518",
    "STGCN": "#54A24B",
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

for axis, metric in zip(axes, metrics):
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        x_values = [row["run"] for row in model_rows]
        y_values = [row[metric] for row in model_rows]
        color = colors[model]
        axis.scatter(
            x_values,
            y_values,
            s=58,
            alpha=0.85,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            label=model,
            zorder=3,
        )
        mean_value = sum(y_values) / len(y_values)
        axis.axhline(mean_value, color=color, linestyle="--", linewidth=1, alpha=0.55)

    axis.set_title(metric.upper())
    axis.set_xlabel("Repeated run")
    axis.set_xticks(range(1, len(seeds) + 1))
    axis.grid(axis="y", linestyle=":", alpha=0.35)

axes[0].set_ylabel("Test metric")
axes[-1].legend(frameon=False, loc="best")
fig.suptitle(f"{dataset} / {target} / 00:00-04:00 baseline repetitions")
fig.tight_layout()

png_path = plot_dir / "repeat_scatter.png"
pdf_path = plot_dir / "repeat_scatter.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Metrics: {metrics_path}")
print(f"Scatter: {png_path}")
print(f"Scatter: {pdf_path}")
PY

echo
echo "All repeated experiments completed."
