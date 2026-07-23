#!/usr/bin/env bash
# ============================================================================
# Run full model training (DCBAM, 5 seeds: 41-45)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/congwei/miniconda3/envs/ts-satfire-fixed/bin/python"

export TS_SATFIRE_DATA_ROOT="${TS_SATFIRE_DATA_ROOT:-}"
export SWIN_PRETRAINED_PATH="${SWIN_PRETRAINED_PATH:-}"
export SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-}"

echo "=================================="
echo "Full Model Training (DCBAM)"
echo "=================================="
echo "Data Root:      ${TS_SATFIRE_DATA_ROOT:-NOT SET}"
echo "Pretrained:     ${SWIN_PRETRAINED_PATH:-NOT SET}"
echo "Sample Manifest: ${SAMPLE_MANIFEST:-NOT SET}"
echo "Config:         configs/full_model.yaml"
echo "Seeds:          41 42 43 44 45"
echo "=================================="

for seed in 41 42 43 44 45; do
    echo ""
    echo "--- Seed $seed ---"
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --execute
    echo "--- Seed $seed completed ---"
done

echo ""
echo "Full model training complete. Check results/training_runs/full_model_corrected_loss/"
