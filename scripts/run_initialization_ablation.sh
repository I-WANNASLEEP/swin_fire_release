#!/usr/bin/env bash
# ============================================================================
# Initialization Ablation: pretrained vs random init × 3 seeds each
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/congwei/miniconda3/envs/ts-satfire-fixed/bin/python"

export TS_SATFIRE_DATA_ROOT="${TS_SATFIRE_DATA_ROOT:-}"
export SWIN_PRETRAINED_PATH="${SWIN_PRETRAINED_PATH:-}"
export SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-}"

SEEDS=(41 42 43)
BASE_OUTPUT="results/training_runs/initialization_ablation_corrected_loss"

echo "=================================="
echo "Initialization Ablation"
echo "=================================="

# Variant 1: Pretrained (cross_modal_rgb_mean_extension)
echo ""
echo ">>> Variant 1: Pretrained (3 seeds)"
for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed ..."
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --override-output-root "${BASE_OUTPUT}/pretrained" \
        --execute
done

# Variant 2: Random init (pass empty pretrained path)
echo ""
echo ">>> Variant 2: Random Init (3 seeds)"
for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed ..."
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --override-output-root "${BASE_OUTPUT}/random_init" \
        --override-pretrained "" \
        --execute
done

echo ""
echo "Initialization ablation complete."
