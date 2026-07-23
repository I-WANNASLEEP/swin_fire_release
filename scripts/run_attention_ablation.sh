#!/usr/bin/env bash
# ============================================================================
# Attention Ablation: none, se, cbam, dcbam × 5 seeds each
# Uses configs/full_model.yaml with --override-attention
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/congwei/miniconda3/envs/ts-satfire-fixed/bin/python"

export TS_SATFIRE_DATA_ROOT="${TS_SATFIRE_DATA_ROOT:-}"
export SWIN_PRETRAINED_PATH="${SWIN_PRETRAINED_PATH:-}"
export SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-}"

SEEDS=(41 42 43 44 45)
ATTENTIONS=(none se cbam dcbam)

echo "=================================="
echo "Attention Ablation"
echo "=================================="
echo "Data Root:      ${TS_SATFIRE_DATA_ROOT:-NOT SET}"
echo "Pretrained:     ${SWIN_PRETRAINED_PATH:-NOT SET}"
echo "Attention variants: ${ATTENTIONS[*]}"
echo "Seeds per variant:  ${SEEDS[*]}"
echo "Output:         results/training_runs/attention_ablation_corrected_loss/"
echo "=================================="

for attn in "${ATTENTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "=== Attention: $attn | Seed: $seed ==="
        "$PYTHON" scripts/train.py \
            --config configs/full_model.yaml \
            --seed "$seed" \
            --override-attention "$attn" \
            --override-output-root "results/training_runs/attention_ablation_corrected_loss" \
            --execute
        echo "=== Done: $attn / seed $seed ==="
    done
done

echo ""
echo "Attention ablation complete."
