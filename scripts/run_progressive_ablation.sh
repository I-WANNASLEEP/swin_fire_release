#!/usr/bin/env bash
# Manuscript-matched progressive ablation: Models A-D, five independent seeds.
#
# A: pretraining + DCBAM + progressive Copy-Paste + cosine restarts + Hybrid Loss
# B: A without Copy-Paste
# C: B without learning-rate optimization (constant LR)
# D: C with masked Cross-Entropy replacing corrected Masked Hybrid Loss
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
BASE_OUTPUT="${PROGRESSIVE_ABLATION_OUTPUT:-results/training_runs/progressive_ablation_corrected_protocol}"
MAX_EPOCHS="${PAPER_MAX_EPOCHS:-100}"
read -r -a SEEDS <<< "${ABLATION_SEEDS:-41 42 43 44 45}"

require_path() {
    local variable_name="$1"
    local value="${!variable_name:-}"
    if [[ -z "$value" || ! -e "$value" ]]; then
        echo "ERROR: $variable_name must point to an existing path; received '${value:-unset}'." >&2
        exit 2
    fi
}

require_path TS_SATFIRE_DATA_ROOT
require_path SWIN_PRETRAINED_PATH
require_path SAMPLE_MANIFEST
command -v "$PYTHON" >/dev/null
"$PYTHON" scripts/materialize_splits.py --check

if [[ "${ALLOW_DIRTY_TRACKED:-0}" != "1" ]] && ! git diff --quiet --exit-code -- .; then
    echo "ERROR: tracked files are modified. Commit the approved protocol first, or use a clean worktree." >&2
    exit 2
fi
if [[ -e "$BASE_OUTPUT" ]]; then
    echo "ERROR: refusing to mix runs in existing output root: $BASE_OUTPUT" >&2
    exit 2
fi

run_variant() {
    local phase="$1"
    local variant="$2"
    local seed="$3"
    shift 3
    local mode_flag="--$phase"
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --override-output-root "${BASE_OUTPUT}/${variant}" \
        --override-max-epochs "$MAX_EPOCHS" \
        "$@" \
        "$mode_flag"
}

echo "Progressive ablation protocol"
echo "Seeds: ${SEEDS[*]}"
echo "Epochs: $MAX_EPOCHS"
echo "Base output: $BASE_OUTPUT"

# Validate every realized run before the first output directory is created.
for seed in "${SEEDS[@]}"; do
    run_variant check model_a_full "$seed"
    run_variant check model_b_without_copy_paste "$seed" --no-copy-paste
    run_variant check model_c_without_lr_optimization "$seed" \
        --no-copy-paste --override-scheduler constant
    run_variant check model_d_ce_only "$seed" \
        --no-copy-paste --override-scheduler constant \
        --override-loss-type masked_cross_entropy
done

for seed in "${SEEDS[@]}"; do
    echo "=== Model A / seed $seed ==="
    run_variant execute model_a_full "$seed"
done
for seed in "${SEEDS[@]}"; do
    echo "=== Model B / seed $seed ==="
    run_variant execute model_b_without_copy_paste "$seed" --no-copy-paste
done
for seed in "${SEEDS[@]}"; do
    echo "=== Model C / seed $seed ==="
    run_variant execute model_c_without_lr_optimization "$seed" \
        --no-copy-paste --override-scheduler constant
done
for seed in "${SEEDS[@]}"; do
    echo "=== Model D / seed $seed ==="
    run_variant execute model_d_ce_only "$seed" \
        --no-copy-paste --override-scheduler constant \
        --override-loss-type masked_cross_entropy
done

echo "Progressive ablation complete: $BASE_OUTPUT"
