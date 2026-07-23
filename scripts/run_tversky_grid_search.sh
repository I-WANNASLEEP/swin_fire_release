#!/usr/bin/env bash
# ============================================================================
# Tversky (alpha, beta) Grid Search on Validation Set
#
# Candidates: (0.5, 0.5), (0.4, 0.6), (0.3, 0.7)
# Each candidate: train with 1 seed (or multiple) -> evaluate on validation
# -> report best F1. Final (alpha, beta) chosen from validation only.
#
# This script runs a quick training for each candidate. For final selection
# you may want more seeds; increase SEEDS array below.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/congwei/miniconda3/envs/ts-satfire-fixed/bin/python"

export TS_SATFIRE_DATA_ROOT="${TS_SATFIRE_DATA_ROOT:-}"
export SWIN_PRETRAINED_PATH="${SWIN_PRETRAINED_PATH:-}"
export SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-}"

SEEDS=(41)  # Use more seeds for final selection
BASE_OUTPUT="results/training_runs/tversky_grid_search"

# (alpha, beta) candidates - alpha=FP weight, beta=FN weight
# (0.5, 0.5): symmetric (Dice-like)
# (0.4, 0.6): moderate recall emphasis  
# (0.3, 0.7): strong recall emphasis (penalize missed fires)
CANDIDATES=(
    "0.5 0.5"
    "0.4 0.6"
    "0.3 0.7"
)

echo "=================================="
echo "Tversky (alpha, beta) Grid Search"
echo "=================================="
echo "Candidates:"
for cand in "${CANDIDATES[@]}"; do
    read -r a b <<< "$cand"
    echo "  alpha=$a beta=$b"
done
echo "Seeds per candidate: ${SEEDS[*]}"
echo "REMINDER: Choose from validation metrics only. Do NOT look at test."
echo "=================================="

for candidate in "${CANDIDATES[@]}"; do
    read -r alpha beta <<< "$candidate"
    variant_name="alpha_${alpha}_beta_${beta}"
    echo ""
    echo ">>> Candidate: alpha=$alpha beta=$beta"
    for seed in "${SEEDS[@]}"; do
        echo "  Seed $seed ..."
        "$PYTHON" scripts/train.py \
            --config configs/full_model.yaml \
            --seed "$seed" \
            --override-tversky-alpha "$alpha" \
            --override-tversky-beta "$beta" \
            --override-output-root "${BASE_OUTPUT}/${variant_name}" \
            --execute
        echo "  Done seed $seed"
    done
done

echo ""
echo "Tversky grid search complete."
echo "Now inspect validation metrics in each variant's epoch_metrics.jsonl"
echo "and select the (alpha, beta) with the highest val_f1_score."
echo ""
echo "To find: for each variant, grep the best validation F1:"
for candidate in "${CANDIDATES[@]}"; do
    read -r a b <<< "$candidate"
    echo "  grep val_best_f1 ${BASE_OUTPUT}/alpha_${a}_beta_${b}/seed_*/epoch_metrics.jsonl | sort -t: -k2 -nr | head -3"
done
