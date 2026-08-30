#!/usr/bin/env bash
set -euo pipefail

models=(CMuST AGCRN ASTGCN GWN STGCN STGODE STTN)

for model_name in "${models[@]}"; do
  if [[ "$model_name" == "CMuST" ]]; then
    DATASETS=SIP bash "$(dirname "$0")/_run_one_baseline.sh" "$model_name" "$@"
  else
    bash "$(dirname "$0")/_run_one_baseline.sh" "$model_name" "$@"
  fi
done
