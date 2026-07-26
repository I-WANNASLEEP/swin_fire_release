#!/usr/bin/env bash
# Validation-only Tversky selection under one locked five-seed protocol.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
BASE_OUTPUT="${TVERSKY_OUTPUT:-results/training_runs/tversky_grid_corrected_protocol}"
MAX_EPOCHS="${PAPER_MAX_EPOCHS:-100}"
read -r -a SEEDS <<< "${ABLATION_SEEDS:-41 42 43 44 45}"
CANDIDATES=("0.5 0.5" "0.45 0.55" "0.55 0.45")

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
    echo "ERROR: tracked files are modified. Use a clean committed worktree." >&2
    exit 2
fi
if [[ -e "$BASE_OUTPUT" ]]; then
    echo "ERROR: refusing existing output root: $BASE_OUTPUT" >&2
    exit 2
fi

run_candidate() {
    local phase="$1"
    local alpha="$2"
    local beta="$3"
    local seed="$4"
    "$PYTHON" scripts/train.py \
        --config configs/full_model.yaml \
        --seed "$seed" \
        --override-tversky-alpha "$alpha" \
        --override-tversky-beta "$beta" \
        --override-max-epochs "$MAX_EPOCHS" \
        --override-output-root "${BASE_OUTPUT}/alpha_${alpha}_beta_${beta}" \
        "--$phase"
}

echo "Tversky validation grid: ${CANDIDATES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "The test set must remain unopened during candidate selection."

for candidate in "${CANDIDATES[@]}"; do
    read -r alpha beta <<< "$candidate"
    for seed in "${SEEDS[@]}"; do
        run_candidate check "$alpha" "$beta" "$seed"
    done
done

for candidate in "${CANDIDATES[@]}"; do
    read -r alpha beta <<< "$candidate"
    for seed in "${SEEDS[@]}"; do
        echo "=== alpha=$alpha beta=$beta / seed $seed ==="
        run_candidate execute "$alpha" "$beta" "$seed"
    done
done

echo "Tversky grid complete: $BASE_OUTPUT"
