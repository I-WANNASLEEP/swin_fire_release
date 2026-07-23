#!/usr/bin/env bash
# ============================================================================
# Progressive Ablation: Models A, B, C, D × 5 seeds each
#
# Model A: Full model (DCBAM, copy-paste, cosine_restart_decay, masked_hybrid)
# Model B: No copy-paste (DCBAM, cosine_restart_decay, masked_hybrid)
# Model C: No copy-paste + step scheduler (DCBAM, step, masked_hybrid)
# Model D: CE-only (DCBAM, step, masked_cross_entropy, no copy-paste)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/congwei/miniconda3/envs/ts-satfire-fixed/bin/python"

export TS_SATFIRE_DATA_ROOT="${TS_SATFIRE_DATA_ROOT:-}"
export SWIN_PRETRAINED_PATH="${SWIN_PRETRAINED_PATH:-}"
export SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-}"

SEEDS=(41 42 43 44 45)
BASE_OUTPUT="results/training_runs/progressive_ablation_corrected_protocol"

echo "=================================="
echo "Progressive Ablation (Models A-D)"
echo "=================================="

# Model A: Full model
echo ""
echo ">>> Model A: Full (copy-paste, cosine_restart_decay, masked_hybrid)"
for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed ..."
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --override-output-root "${BASE_OUTPUT}/model_a_full" \
        --execute
done

# Model B: Without copy-paste
echo ""
echo ">>> Model B: No copy-paste (cosine_restart_decay, masked_hybrid)"
for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed ..."
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --override-output-root "${BASE_OUTPUT}/model_b_no_copy_paste" \
        --no-copy-paste \
        --execute
done

# Model C: No copy-paste + step scheduler
echo ""
echo ">>> Model C: No copy-paste + step scheduler (step LR)"
for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed ..."
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --override-scheduler step \
        --override-output-root "${BASE_OUTPUT}/model_c_step_scheduler" \
        --no-copy-paste \
        --execute
done

# Model D: CE-only (masked_cross_entropy, step scheduler, no copy-paste)
echo ""
echo ">>> Model D: CE-only (step LR, masked_cross_entropy)"
for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed ..."
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --override-scheduler step \
        --override-loss-type masked_cross_entropy \
        --override-output-root "${BASE_OUTPUT}/model_d_ce_only" \
        --no-copy-paste \
        --execute
done

echo ""
echo "Progressive ablation complete."
