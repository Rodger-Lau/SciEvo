set -euo pipefail

mkdir -p outputs/BrainAI/SIP

scales=(0.8)

for scale in "${scales[@]}"; do
  tag="fewshot_${scale//./_}"
  python -u main_ST.py \
    --dataset SIP \
    --num_nodes 108 \
    --tod_size 288 \
    --seed 2025 \
    --experiment_name "$tag" \
    --few_shot_scale "$scale" \
    > "outputs/BrainAI/SIP/${tag}.txt" 2>&1
done