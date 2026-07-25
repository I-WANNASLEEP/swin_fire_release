#!/usr/bin/env bash
# Architecture Baselines: swin_convlstm, swinunetr3d, unet3d × 5 seeds each
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"

if ! git diff --quiet -- || ! git diff --cached --quiet --; then
    echo "Refusing a paper baseline from a dirty tracked source tree." >&2
    echo "Commit or restore the exact source changes before training so W&B's Git SHA identifies the executed code." >&2
    exit 2
fi

export TS_SATFIRE_DATA_ROOT="${TS_SATFIRE_DATA_ROOT:-}"
export SWIN_PRETRAINED_PATH="${SWIN_PRETRAINED_PATH:-}"
export SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-}"

# The paper protocol uses five independent seeds.  A single-seed diagnostic can
# be launched explicitly with ARCH_BASELINE_SEEDS="42".
read -r -a SEEDS <<< "${ARCH_BASELINE_SEEDS:-41 42 43 44 45}"
BASE_OUTPUT="${ARCH_BASELINE_OUTPUT:-results/training_runs/architecture_baselines_numerical_fix_v2}"
failures=()

echo "Architecture Baselines"

# Note: swinunetr3d and unet3d use the legacy trainer with different model names.
# They are launched directly via train_models_spatial_temp.py

for model_name in swinunetr3d unet3d; do
    if [[ "$model_name" == "swinunetr3d" ]]; then
        model_attention="v2"
    else
        model_attention="none"
    fi
    for seed in "${SEEDS[@]}"; do
        run_index=$((seed - 41))
        run_dir="${PROJECT_DIR}/${BASE_OUTPUT}/${model_name}/seed_${seed}"
        if [[ -e "$run_dir/epoch_metrics.jsonl" ||
              -e "$run_dir/wandb_run.json" ||
              -e "$run_dir/startup_provenance.json" ]]; then
            failures+=("$model_name/seed_$seed (existing run artifacts)")
            echo "=== Refusing to mix a new run with existing artifacts: $run_dir ===" >&2
            continue
        fi
        mkdir -p "$run_dir"
        echo ""
        echo "=== Model: $model_name | Seed: $seed (run $run_index) ==="
        if "$PYTHON" train_models_spatial_temp.py \
                -m "$model_name" -mode af -b 1 \
                -r "$run_index" -lr 0.0001 -av "$model_attention" \
                -nh 4 -ed 96 -nc 8 -ts 10 -it 3 \
                --max-epochs 100 -patience 15 -grad_clip 1.0 \
                -scheduler cosine_restart_decay \
                --data-root "$TS_SATFIRE_DATA_ROOT" \
                --allow-random-init \
                --output-dir "$run_dir" \
                --loss-type masked_hybrid \
                --wandb-mode online \
                --wandb-project swinfire_jei_resubmission_v2 \
                --wandb-require-final-metrics; then
            echo "=== Done: $model_name / seed $seed ==="
        else
            status=$?
            failures+=("$model_name/seed_$seed (exit $status)")
            echo "=== Failed: $model_name / seed $seed (exit $status); continuing to the next baseline ===" >&2
        fi
    done
done

echo ""
if (( ${#failures[@]} > 0 )); then
    echo "Architecture baseline failures:" >&2
    printf '  - %s\n' "${failures[@]}" >&2
    exit 1
fi

echo "Architecture baselines complete. (swin_convlstm = full model experiment)"
