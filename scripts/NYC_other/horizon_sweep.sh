#!/usr/bin/env bash
set -euo pipefail

DATASET="NYC"
NUM_NODES=206
TOD_SIZE=48
SEED=2025
DATA_ROOT="/root/autodl-tmp/data/h24"
OUTPUT_DIR="outputs/BrainAI/${DATASET}"

mkdir -p "${OUTPUT_DIR}"

HORIZONS=(18 24)

for horizon in "${HORIZONS[@]}"; do
  exp_name="horizon_${horizon}"
  python -u main_ST.py \
    --dataset "${DATASET}" \
    --num_nodes "${NUM_NODES}" \
    --tod_size "${TOD_SIZE}" \
    --seed "${SEED}" \
    --data_root "${DATA_ROOT}" \
    --experiment_name "${exp_name}" \
    --input_len 12 \
    --output_len "${horizon}" \
    > "${OUTPUT_DIR}/${exp_name}.txt" 2>&1
done
