# Reproducible TS-SatFire SwinConvLSTM for JEI resubmission

This repository contains the corrected training, evaluation, and evidence
pipeline for 8-channel, 10-frame active-fire segmentation. The goal is not to
preserve a historical score; it is to make every manuscript number traceable to
one locked data split, one source revision, one independent seed, one raw metric
record, and one W&B run.

No result produced with the former Hybrid Loss is a final result. The historical
checkpoint with validation F1 around 0.82 remains useful for qualitative
debugging, but it lacks the immutable data and multi-seed provenance required
for the revised paper.

## Method implemented by the code

The primary model receives `[B, 8, 10, 256, 256]` input in this fixed channel
order:

1. `I1_day`
2. `I2_day`
3. `I3_day`
4. `I4_day`
5. `I5_day`
6. `M11_day`
7. `I4_night`
8. `I5_night`

Each frame is encoded by a 2D MONAI SwinUNETR. The selected attention module is
applied to the encoder output before ConvLSTM temporal aggregation, and a
two-class segmentation head returns `[B, 2, 10, 256, 256]`. The manuscript must
describe this actual placement; the current code does not place DCBAM in decoder
skip connections.

Active-fire labels use channel index 2. Validity and class semantics are:

```text
valid pixel:   target != -1
fire pixel:    target > 0
fire prob.:    softmax(logits, dim=1)[:, 1]
```

`losses/masked_hybrid_loss.py` excludes invalid pixels before every reduction and
computes:

```text
0.4 × Tversky + 0.3 × Focal + 0.3 × Cross-Entropy
```

The tests cover ignore-mask invariance, zero ignored-pixel gradient, all-invalid
crops, extreme class ratios, non-binary positive encodings, finite-value
validation, CPU, CUDA, and AMP.

## Locked data and selection protocol

This project locks 125 training events, 13 validation events, and **24 held-out
test units**. At the user's direction, the 24 `US_2021_*` NPY pairs supplied in
`dataset_test/` are authoritative and match `splits/test_event_ids.txt` exactly
after removing `af_`.

This is an explicit project-level test definition: the pinned upstream generator
uses 17 named-fire labels, whereas the revised study reports the 24 supplied
`US_2021_*` units. The manuscript must state that distinction instead of calling
the lists identical. Each supplied identifier is the unit for per-event metrics
and event-level Bootstrap. Do not reintroduce directory-order selection or a
first-ten subset.

- Training: parameter optimization only.
- Validation: Tversky selection, threshold selection, checkpoint selection, and
  early stopping.
- Test: one final evaluation after model and threshold are frozen.
- Allowed validation thresholds: `0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65`.
- Checkpoint eligibility starts at epoch 51.
- Early-stopping counting starts at epoch 101.
- One run retains exactly one atomically replaced best checkpoint.
- Repetitions are independent seeds, never epochs.
- Report `mean ± sample standard deviation` across seeds.
- Bootstrap confidence intervals over held-out fire events, never epochs.

The provided per-event test arrays are finite and structurally valid, but they
do not contain a window-level timestamp manifest. Until event/window timestamps
and file hashes are frozen, frame numbers 1–10 are sequence indices rather than
verified acquisition timestamps.

## Environment

Use the existing environment; do not recreate it.

Local macOS:

```bash
conda activate ts-satfire
cd /path/to/swin_fire_released
export PYTHON=/opt/miniconda3/envs/ts-satfire/bin/python
```

The remote GPU environment is commonly named `swin`:

```bash
conda activate swin
cd /path/to/swin_fire_released
export PYTHON=python
```

`requirements.txt` is a compatibility record, not an instruction to replace the
working Conda environment.

Before training, set:

```bash
export TS_SATFIRE_DATA_ROOT=/absolute/path/to/processed
export SWIN_PRETRAINED_PATH=/absolute/path/to/pretrained_checkpoint.pth
export SAMPLE_MANIFEST=/absolute/path/to/sample_manifest.csv
```

The sample manifest must map every generated window to `event_id` and `split`.
For final experiments it should also contain the window timestamps and should be
accompanied by SHA256 hashes of every aggregated NPY file.

## Training protocol

The paper configuration in `configs/full_model.yaml` uses:

```text
batch size                    1
learning rate                 1e-4
epochs                        100
seeds                         41, 42, 43, 44, 45
gradient clipping             1.0
cosine restart T_0            10 epochs
T_mult                        1
peak decay per restart        0.99
minimum learning rate         1e-7
scheduler step unit           epoch
```

The scheduler is stepped once after validation. Its trajectory therefore does
not depend on batch size, number of batches, or skipped samples. These exact
restart fields are recorded in the run manifest, startup provenance, checkpoint,
and W&B configuration.

Run a non-mutating preflight first:

```bash
"$PYTHON" scripts/train.py \
  --config configs/full_model.yaml \
  --seed 41 \
  --check
```

Run one seed only after that check succeeds:

```bash
"$PYTHON" scripts/train.py \
  --config configs/full_model.yaml \
  --seed 41 \
  --execute
```

The launcher refuses to append to a non-empty seed directory. Paper launchers
also refuse tracked source changes by default; use a clean committed worktree.

## Manuscript-matched ablations

### Progressive ablation

| Variant | Pretraining | DCBAM | Copy-Paste | LR schedule | Loss |
|---|---:|---:|---:|---|---|
| Model A | yes | yes | progressive | epoch cosine restarts | corrected Masked Hybrid |
| Model B | yes | yes | disabled | epoch cosine restarts | corrected Masked Hybrid |
| Model C | yes | yes | disabled | constant LR | corrected Masked Hybrid |
| Model D | yes | yes | disabled | constant LR | masked Cross-Entropy |

“Remove learning-rate optimization” is implemented literally as constant LR.
Replacing cosine restarts with StepLR would introduce a new scheduler rather
than remove the component.

```bash
bash scripts/run_progressive_ablation.sh
```

### Attention ablation

The current manuscript table compares `none`, `se`, and `dcbam`. CBAM is not
silently added to that table.

```bash
bash scripts/run_attention_ablation.sh
```

Both launchers default to five seeds and 100 epochs, preflight every realized
run before starting the first GPU job, create a separate output root per
variant, and refuse an existing experiment root. Environment variables
`ABLATION_SEEDS` and `PAPER_MAX_EPOCHS` are for explicit smoke tests only; runs
using them are not automatically paper-admissible.

Other controlled experiments:

```bash
bash scripts/run_tversky_grid_search.sh
bash scripts/run_initialization_ablation.sh
bash scripts/run_architecture_baselines.sh
```

## Epoch-0 initialization audit

Every SwinConvLSTM run now performs the same validation pass before any optimizer
step, whether it loads pretraining or uses module-native random initialization.
The audit is written to:

```text
epoch0_validation_audit.json
audit/epoch0/* in W&B
```

Epoch-0 F1 is a diagnostic of initial output calibration, not a test of whether
pretrained weights loaded. The 8-channel patch embedding, temporal module, and
segmentation head can yield poorly calibrated logits even when the checkpoint
load report confirms that pretrained encoder layers were loaded. Pretraining
must be judged by paired, same-code, same-seed learning curves and final
validation-selected checkpoints.

## Ten-frame test visualization

The evaluator requires a checkpoint threshold frozen on validation. It renders
all ten frames, per-frame metrics, raw probabilities, a compact PNG/GIF, and two
five-frame diagnostic pages with the same four columns as the manuscript
example: input, ground truth, binary prediction, and TP/FP/FN overlay. Only
pixels belonging to the ground-truth or predicted fire regions are recolored;
all other pixels retain the original displayed grayscale. In accordance with
the requested color convention:

- TP: bright green
- FP: red
- FN: dark green with a bright-green outline

The two green classes remain distinguishable in the legend and by the FN
outline. Use `--fn-color blue` only when an exact reproduction of the older
TP-green/FP-red/FN-blue reference figure is required; green is the default.

```bash
"$PYTHON" scripts/visualize_test_sequence.py \
  --checkpoint /absolute/path/to/frozen_checkpoint.pth \
  --test-dir dataset_test \
  --event-id US_2021_AZ3345510938920210616 \
  --window-index 0 \
  --display-channel I4_day \
  --fn-color green
```

The output directory contains:

```text
sequence_diagnostic_frames_01_05.png
sequence_diagnostic_frames_06_10.png
sequence_tp_fp_fn.png
sequence_tp_fp_fn.gif
sequence_metrics.json
fire_probabilities.npy
```

Do not search thresholds on `dataset_test/`. `--threshold` is accepted only to
restate a value that was already frozen before test evaluation.

## DCBAM attention heatmaps

The attention visualizer uses forward hooks to capture the trained
`attention.conv_spatial` output for each frame and applies the same sigmoid used
in the model. It exports a PNG, GIF, the exact spatial tensors, fire
probabilities, and 96 latent feature-channel weights.

```bash
"$PYTHON" scripts/visualize_attention_heatmap.py \
  --checkpoint /absolute/path/to/frozen_dcbam_checkpoint.pth \
  --test-dir dataset_test \
  --event-id US_2021_AZ3345510938920210616 \
  --window-index 0
```

`attention_heatmap.png` and the GIF place only the relative heatmap over the
original grayscale input. Post-Sigmoid attention is normalized with the
sequence-wide P2--P98 range. Relative values below `0.9` are fully transparent;
values from `0.9` to `1.0` transition from yellow to red while opacity increases
linearly from `0.3` to `1.0`. The NPZ preserves both pre- and post-Sigmoid
absolute tensors, because transparent relative pixels can still have high
absolute attention.

The 96 channel weights are latent Swin feature channels, not the eight physical
input bands. Attention intensity is a model-internal weight, not a causal
explanation or an independently validated localization score.

Use the aligned comparison exporter to place the same ten input frames, fire
labels, frozen-threshold predictions, relative attention, and categorical
fire-attention overlap in one layout:

```bash
"$PYTHON" scripts/visualize_fire_attention_alignment.py \
  --checkpoint /absolute/path/to/frozen_dcbam_checkpoint.pth \
  --test-dir dataset_test \
  --event-id US_2021_AZ3345510938920210616 \
  --window-index 0
```

It exports two five-frame comparison pages, a ten-frame GIF, per-frame overlap
CSV/JSON, a temporal overlap-metric plot, and the exact aligned arrays. Fire
coverage measures the fraction of ground-truth fire inside the displayed
relative-attention region; attention precision measures the fraction of that
attention region that is actually fire. Neither is a segmentation accuracy or
causal-explanation metric.

Historical checkpoints may be used to verify visualization compatibility and
to produce clearly labeled qualitative examples. They must not be reported as
post-fix paper results unless they were trained under the locked split, loss,
scheduler, threshold-selection, and provenance protocol.

## W&B and reproducible analysis

Final paper runs must use online W&B project
[`15145202826-1/swinfire_jei_resubmission_v2`](https://wandb.ai/15145202826-1/swinfire_jei_resubmission_v2).
Each run also writes:

```text
resolved_config.json
run_manifest.json
startup_provenance.json
epoch0_validation_audit.json
epoch_metrics.jsonl
wandb_run.json
```

Export a protocol-aware audit of the full project:

```bash
"$PYTHON" scripts/analyze_wandb_runs.py \
  --output-dir results/analysis/wandb_YYYY-MM-DD
```

The export separates runs by source commit, initialization, loss, scheduler,
planned epochs, checkpoint policy, threshold grid, and actual LR trace. It does
not combine runs merely because their names end in the same alpha/beta pair.

Generate manuscript tables and curves only from raw records:

```bash
"$PYTHON" scripts/reproduce_all_tables.py \
  --input results/raw_metrics \
  --output results
"$PYTHON" scripts/reproduce_training_curves.py \
  --input results/training_runs \
  --output results
```

## Verification

On macOS:

```bash
"$PYTHON" -m unittest \
  losses.test_masked_hybrid_loss \
  test_scheduler_protocol \
  test_visualization_protocol \
  test_initialization_protocol \
  test_3d_baselines -v
bash -n scripts/run_progressive_ablation.sh
bash -n scripts/run_attention_ablation.sh
```

CUDA/AMP tests skip on macOS and must be repeated on the remote GPU:

```bash
conda activate swin
python -m unittest losses.test_masked_hybrid_loss test_3d_baselines -v
```

For the complete evidence chain, see
[`docs/reproduction.md`](docs/reproduction.md),
[`docs/dataset_versions.md`](docs/dataset_versions.md),
[`docs/airborne_metadata.md`](docs/airborne_metadata.md),
[`docs/retraining_manifest.md`](docs/retraining_manifest.md), and
[`docs/manuscript_revision_outline.md`](docs/manuscript_revision_outline.md).
