set -euo pipefail

mkdir -p outputs/BrainAI/CHI

scales=(0.2 0.4 0.6 0.8)

for scale in "${scales[@]}"; do
  tag="fewshot_${scale//./_}"
  python -u main_ST.py \
    --dataset CHI \
    --num_nodes 220 \
    --tod_size 48 \
    --seed 2025 \
    --experiment_name "$tag" \
    --few_shot_scale "$scale" \
    > "outputs/BrainAI/CHI/${tag}.txt" 2>&1
done