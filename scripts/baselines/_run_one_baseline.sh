#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL_NAME [extra main_ST.py args...]" >&2
  exit 1
fi

model_name="$1"
shift

seed="${SEED:-2025}"
source_epochs="${SOURCE_EPOCHS:-10}"
target_epochs="${TARGET_EPOCHS:-50}"
batch_size="${BATCH_SIZE:-16}"
target_source_idx="${TARGET_SOURCE_IDX:-0}"
target_task_idx="${TARGET_TASK_IDX:-0}"

datasets=(${DATASETS:-CHI NYC SIP})

for idx in "${!datasets[@]}"; do
  dataset="${datasets[$idx]}"
  case "$dataset" in
    CHI)
      nodes=220
      tod_size=48
      ;;
    NYC)
      nodes=206
      tod_size=48
      ;;
    SIP)
      nodes=108
      tod_size=288
      ;;
    *)
      echo "Unsupported DATASETS entry: $dataset. Use CHI, NYC, or SIP." >&2
      exit 1
      ;;
  esac
  exp_name="baseline_${model_name}"

  mkdir -p "outputs/BrainAI/${dataset}"

  python -u main_ST.py \
    --dataset "$dataset" \
    --num_nodes "$nodes" \
    --tod_size "$tod_size" \
    --seed "$seed" \
    --target_source_idx "$target_source_idx" \
    --target_task_idx "$target_task_idx" \
    --training_strategy random_baseline \
    --model_name "$model_name" \
    --experiment_name "$exp_name" \
    --batch_size "$batch_size" \
    --baseline_source_epochs "$source_epochs" \
    --baseline_target_epochs "$target_epochs" \
    "$@" \
    > "outputs/BrainAI/${dataset}/${exp_name}.txt" 2>&1
done
