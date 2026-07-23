#!/usr/bin/env bash
# ============================================================================
# Architecture Baselines: swin_convlstm, swinunetr3d, unet3d × 5 seeds each
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/congwei/miniconda3/envs/ts-satfire-fixed/bin/python"

export TS_SATFIRE_DATA_ROOT="${TS_SATFIRE_DATA_ROOT:-}"
export SWIN_PRETRAINED_PATH="${SWIN_PRETRAINED_PATH:-}"
export SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-}"

SEEDS=(41 42 43 44 45)
BASE_OUTPUT="results/training_runs/architecture_baselines_corrected_protocol"

echo "=================================="
echo "Architecture Baselines"
echo "=================================="

# Note: swinunetr3d and unet3d use the legacy trainer with different model names.
# They are launched directly via train_models_spatial_temp.py

for model_name in swinunetr3d unet3d; do
    for seed in "${SEEDS[@]}"; do
        run_index=$((seed - 41))
        run_dir="${PROJECT_DIR}/${BASE_OUTPUT}/${model_name}/seed_${seed}"
        mkdir -p "$run_dir"
        echo ""
        echo "=== Model: $model_name | Seed: $seed (run $run_index) ==="
        "$PYTHON" train_models_spatial_temp.py \
            -m "$model_name" -mode af -b 1 \
            -r "$run_index" -lr 0.0001 -av none \
            -nh 4 -ed 96 -nc 8 -ts 10 -it 3 \
            --max-epochs 100 -patience 15 -grad_clip 1.0 \
            -scheduler cosine_restart_decay \
            --data-root "$TS_SATFIRE_DATA_ROOT" \
            --pretrained-path "" \
            --output-dir "$run_dir" \
            --loss-type masked_hybrid \
            --wandb-mode online \
            --wandb-project swinfire_jei_resubmission_v2 \
            --wandb-require-final-metrics
        echo "=== Done: $model_name / seed $seed ==="
    done
done

echo ""
echo "Architecture baselines complete. (swin_convlstm = full model experiment)"
