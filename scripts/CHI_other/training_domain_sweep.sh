#!/usr/bin/env bash
set -euo pipefail

dataset="CHI"
num_nodes=220
tod_size=48
seed=2025
target_source_idx=0
target_task_idx=0

mkdir -p "outputs/BrainAI/${dataset}"

run_experiment() {
  local exp_name="$1"
  local training_strategy="$2"
  shift 2

  python -u main_ST.py \
    --dataset "$dataset" \
    --num_nodes "$num_nodes" \
    --tod_size "$tod_size" \
    --seed "$seed" \
    --target_source_idx "$target_source_idx" \
    --target_task_idx "$target_task_idx" \
    --training_strategy "$training_strategy" \
    --model_name CMuST \
    --experiment_name "$exp_name" \
    "$@" \
    > "outputs/BrainAI/${dataset}/${exp_name}.txt" 2>&1
}

run_method_pair() {
  local ours_exp_name="$1"
  local backbone_exp_name="$2"
  shift 2

  run_experiment "$ours_exp_name" proposed "$@" &
  local ours_pid=$!

  run_experiment "$backbone_exp_name" random_baseline "$@" &
  local backbone_pid=$!

  local status=0
  wait "$ours_pid" || status=$?
  wait "$backbone_pid" || status=$?
  return "$status"
}

run_method_pair \
  "train_domains_1_TAXIDROP_ours" \
  "train_domains_1_TAXIDROP_CMuST_backbone" \
  --train_source_idxs 2 \
  "$@"

run_method_pair \
  "train_domains_2_TAXIPICK_TAXIDROP_ours" \
  "train_domains_2_TAXIPICK_TAXIDROP_CMuST_backbone" \
  --train_source_idxs 1 2 \
  "$@"
