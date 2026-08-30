set -euo pipefail

mkdir -p outputs/BrainAI/NYC

scales=(0.2 0.4 0.6 0.8)

for scale in "${scales[@]}"; do
  tag="fewshot_${scale//./_}"
  python -u main_ST.py \
    --dataset NYC \
    --num_nodes 206 \
    --tod_size 48 \
    --seed 2025 \
    --experiment_name "$tag" \
    --few_shot_scale "$scale" \
    > "outputs/BrainAI/NYC/${tag}.txt" 2>&1
done