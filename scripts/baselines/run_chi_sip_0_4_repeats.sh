#!/usr/bin/env bash
set -euo pipefail

# Run STGODE, CMuST, and STGCN with five seeds on:
#   CHI / RISK / 00:00-04:00
#   SIP / FLOW / 00:00-04:00
#
# Each dataset uses the generic runner, which launches at most two experiments
# in parallel and skips experiments whose summary.csv already exists.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

max_parallel="${MAX_PARALLEL:-2}"
models="${MODELS:-STGODE CMuST STGCN}"
seeds="${SEEDS:-2025 2026 2027 2028 2029}"
skip_completed="${SKIP_COMPLETED:-1}"
dry_run="${DRY_RUN:-0}"

if ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got $max_parallel." >&2
  exit 1
fi

echo "CHI + SIP repeated baseline experiments"
echo "Models          : $models"
echo "Seeds           : $seeds"
echo "Parallel jobs   : $max_parallel"
echo "Skip completed  : $skip_completed"

for dataset in CHI SIP; do
  echo
  echo "============================================================"
  echo "Starting dataset: $dataset"
  echo "============================================================"

  DATASET="$dataset" \
  MODELS="$models" \
  SEEDS="$seeds" \
  MAX_PARALLEL="$max_parallel" \
  SKIP_COMPLETED="$skip_completed" \
  DRY_RUN="$dry_run" \
    bash "${script_dir}/run_baseline_0_4_repeats.sh" "$@"
done

echo
echo "All CHI + SIP repeated baseline experiments completed."
