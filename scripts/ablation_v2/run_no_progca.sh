#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${DATASET:-}" ]]; then
  exec "${script_dir}/_run_ablation.sh" progca "$@"
fi

read -r -a datasets <<< "${DATASETS:-NYC CHI SIP}"
for dataset in "${datasets[@]}"; do
  DATASET="$dataset" "${script_dir}/_run_ablation.sh" progca "$@"
done
