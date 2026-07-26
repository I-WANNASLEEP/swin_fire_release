#!/usr/bin/env bash
# Manuscript-matched attention ablation: None, SE, and DCBAM.
# All other settings are inherited unchanged from configs/full_model.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
BASE_OUTPUT="${ATTENTION_ABLATION_OUTPUT:-results/training_runs/attention_ablation_corrected_protocol}"
MAX_EPOCHS="${PAPER_MAX_EPOCHS:-100}"
read -r -a SEEDS <<< "${ABLATION_SEEDS:-41 42 43 44 45}"
ATTENTIONS=(none se dcbam)

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
    local attention="$2"
    local seed="$3"
    local mode_flag="--$phase"
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --override-attention "$attention" \
        --override-output-root "${BASE_OUTPUT}/${attention}" \
        --override-max-epochs "$MAX_EPOCHS" \
        "$mode_flag"
}

echo "Attention ablation protocol"
echo "Variants: ${ATTENTIONS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Epochs: $MAX_EPOCHS"
echo "Base output: $BASE_OUTPUT"

# Preflight all 15 runs before starting the first GPU job.
for attention in "${ATTENTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_variant check "$attention" "$seed"
    done
done

for attention in "${ATTENTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=== Attention $attention / seed $seed ==="
        run_variant execute "$attention" "$seed"
    done
done

echo "Attention ablation complete: $BASE_OUTPUT"
