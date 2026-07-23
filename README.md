# Corrected-loss wildfire segmentation revision

This repository is being rebuilt for a reproducible JEI resubmission. It keeps
the SwinConvLSTM research direction and 8-channel TS-SatFire input, but treats
all results produced with the former Hybrid Loss as **historical and invalid for
the corrected experiment**. No revised metric is pre-filled in this repository.

## What is fixed

- `losses/masked_hybrid_loss.py` uses two-class Softmax, masks `-1` pixels
  before every Tversky/Focal/CE reduction, and implements
  `0.4 × Tversky + 0.3 × Focal + 0.3 × CE`.
- `losses/test_masked_hybrid_loss.py` tests ignored-pixel invariance and zero
  gradient, all-invalid crops, extreme foreground ratios, Softmax, CPU/GPU, and
  AMP behavior.
- `splits/` is a locked official active-fire event partition: 125 training,
  13 validation, and 17 test events. It replaces all directory-order or
  “first 10” selection.
- `metadata/` pins the upstream code revision, channel order, normalization
  provenance, preprocessing contract, and dataset-version requirements.
- `scripts/evaluate.py` emits masked, per-event raw metric records;
  `scripts/reproduce_all_tables.py` derives tables/figure data from those raw
  records only. The trainer writes one local JSONL record per epoch, and
  `scripts/reproduce_training_curves.py` derives convergence curves from those
  logs only.

## Environment: use the existing Conda environment

Do not download or recreate an environment. Use the existing `ts-satfire`
environment already on this computer:

```bash
PROJECT=/Users/congwei/Documents/遥感火灾论文/swin_fire_released
PY=/opt/miniconda3/envs/ts-satfire/bin/python
cd "$PROJECT"
"$PY" -m unittest losses.test_masked_hybrid_loss
"$PY" scripts/materialize_splits.py --check
```

`requirements.txt` is a reference compatibility target, not an instruction to
install packages. The revised code makes optional logging/debug/plot packages
non-blocking when the existing environment lacks them.

## Reproduce a corrected experiment

The original data and pretrained checkpoint are intentionally not copied into
this repository. Before training, set the `path/to/...` values in
`configs/full_model.yaml` to your local preprocessed array root, event-window
manifest, and pretrained Swin checkpoint. The window manifest must include
`event_id,split`; do not evaluate aggregate arrays that cannot be traced to an
event.

```bash
"$PY" scripts/train.py --config configs/full_model.yaml --seed 41 --check

for seed in 41 42 43 44 45; do
  "$PY" scripts/train.py --config configs/full_model.yaml --seed "$seed" --execute
done
```

Choose the Tversky `(alpha, beta)` grid and segmentation threshold on validation
events only. Then freeze them and run `scripts/evaluate.py` for every held-out
test event. Generate manuscript assets only after all raw records are present:

```bash
"$PY" scripts/reproduce_all_tables.py --input results/raw_metrics --output results
"$PY" scripts/reproduce_training_curves.py --input results/training_runs --output results
```

The report generator rejects fewer than three independent seeds, reports
mean±standard deviation across seeds, and computes confidence intervals by
bootstrapping held-out fire events—not epochs. Epoch logs are used only for
convergence figures; they are never treated as independent experimental runs.

## Experiment scope

`configs/full_model.yaml`, `configs/attention_ablation.yaml`,
`configs/architecture_baselines.yaml`, and
`configs/initialization_ablation.yaml` define the required controlled studies.
`configs/progressive_ablation.yaml` states the exact Model A–D rerun contract.
The attention comparison is `none`, `se`, `cbam`, and `dcbam`; all non-attention
settings remain fixed. Every prior Hybrid-Loss result (Models A/B/C, attention
ablations, and loss ablations) must be rerun. A CE-only Model D can only be
retained with provenance, but rerunning under this code version is preferred.

`dataset_generate.py` is explicitly `DEPRECATED_NOT_USED_IN_PAPER` because it
selects a directory-derived first-ten test subset. Use the pinned upstream
generator plus `scripts/materialize_splits.py` instead.

## Cross-sensor airborne validation

The old two-channel-with-six-zero-filled image demonstration is not a standalone
cross-sensor experiment. Follow `docs/airborne_metadata.md`, create a manifest
with independent held-out events and calibration/channel-adaptation information,
and validate it with `scripts/evaluate_airborne.py`. Do not report an airborne
metric until that protocol and its raw per-event records exist.

See [docs/reproduction.md](docs/reproduction.md),
[docs/dataset_versions.md](docs/dataset_versions.md), and
[docs/airborne_metadata.md](docs/airborne_metadata.md), and
[docs/retraining_manifest.md](docs/retraining_manifest.md) for the complete
evidence chain. Use
[docs/manuscript_revision_outline.md](docs/manuscript_revision_outline.md) to
rewrite the paper as a JEI imaging-methods article without carrying forward
unsupported historical results.
