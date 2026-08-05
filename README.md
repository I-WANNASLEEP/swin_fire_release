# Reproducible TS-SatFire SwinConvLSTM

This repository contains the corrected training, evaluation, and evidence
pipeline for 8-channel, 10-frame active-fire segmentation. It connects each
experiment to one locked data split, one source revision, one independent seed,
one raw metric record, and one W&B run.

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
test events**. The fixed event identifiers are stored in
`splits/train_event_ids.txt`, `splits/validation_event_ids.txt`, and
`splits/test_event_ids.txt`. Do not select events by directory order or use a
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

The current `dataset_test/` delivery contains 48 NPY files forming 24 complete
image/label pairs and 237 ten-frame windows. A full local scan found finite
`float32` arrays with image shape `[N,8,10,256,256]`, label shape
`[N,3,10,256,256]`, and active-fire values equal to zero or positive radiometric
values rather than a strict `0/1` mask. This structural scan does not replace
the required timestamp/sample manifest or file SHA256 inventory.

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
also refuse staged or unstaged tracked source changes by default; use a clean
committed worktree.

## Manuscript-matched ablations

### Progressive ablation

| Variant | Pretraining | DCBAM | Copy-Paste | LR schedule | Loss |
|---|---:|---:|---:|---|---|
| Model A | no | yes | progressive | epoch cosine restarts | corrected Masked Hybrid |
| Model B | no | yes | disabled | epoch cosine restarts | corrected Masked Hybrid |
| Model C | no | yes | disabled | constant LR | corrected Masked Hybrid |
| Model D | no | yes | disabled | constant LR | masked Cross-Entropy |

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
variant, refuse an existing experiment root, and include the exact variant
(`model_a_full`, `model_b_without_copy_paste`, `model_c_without_lr_optimization`,
`model_d_ce_only`, or `attention_*`) in the run manifest and W&B run name.
Environment variables `ABLATION_SEEDS` and `PAPER_MAX_EPOCHS` are rejected for a
paper run unless `SMOKE_TEST_ONLY=1` is set and a separate smoke-test output root
is supplied.

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
validation-selected checkpoints. Legacy terminal output is not retroactively
present in W&B; only runs created by the current code can be expected to expose
the `audit/epoch0/*` fields.

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
- FN: blue

Blue is the default FN color. The optional `--fn-color green` argument is
retained only for reproducing legacy figures.

```bash
"$PYTHON" scripts/visualize_test_sequence.py \
  --checkpoint /absolute/path/to/frozen_checkpoint.pth \
  --test-dir dataset_test \
  --event-id US_2021_AZ3345510938920210616 \
  --window-index 0 \
  --display-channel I4_day \
  --fn-color blue
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
linearly from `0.3` to `1.0`. For the DCBAM configuration
(`kernel_size=7`, `dilation=2`, `padding=6`), every visualization uses the full
`256×256` tensor for model inference and then synchronously crops the input
base, label, probability, and attention tensors to
`[..., 16:240, 16:240]`. Thus the displayed output is the center `224×224`
region and the model-affected outer boundary cannot form a red frame. This is
visualization-only: it does not alter inference, checkpoint weights, or the
paper's evaluation metrics, and it does not suppress high responses on linear
structures inside the retained center region. The NPZ
preserves both pre- and post-Sigmoid absolute tensors, because transparent
relative pixels can still have high absolute attention.

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

To render all three products for every window in all 24 locked test events,
load the checkpoint only once and use the resumable PNG-only batch exporter:

```bash
"$PYTHON" scripts/visualize_all_test_sequences.py \
  --checkpoint /absolute/path/to/frozen_dcbam_checkpoint.pth \
  --test-dir dataset_test \
  --output-root results/all_test_visualizations
```

The output root is one flat directory containing exactly seven PNG files per
window: three fire-monitoring figures, one ten-frame attention figure, two
aligned comparison figures, and one overlap-metric figure. It contains no
subdirectories, GIF, NPZ, NPY, JSON, or CSV files. If interrupted, rerun the
same command with `--resume`. For a one-window-per-event smoke test, add
`--max-windows-per-event 1`. To regenerate only a specific window, add, for
example:

```bash
  --event-id US_2021_ID4558511544420210705 \
  --window-index 1
```



The export separates runs by source commit, explicit experiment variant,
Copy-Paste state, initialization, loss, scheduler, planned epochs, checkpoint
policy, threshold grid, and actual LR trace. It also exports Epoch-0 audit
fields when the run actually recorded them. It does not combine runs merely
because their names end in the same alpha/beta pair.

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
  datasets.test_label_diagnostics \
  losses.test_masked_hybrid_loss \
  test_initialization_protocol \
  test_training_protocol \
  test_3d_baselines -v
bash -n scripts/run_progressive_ablation.sh
bash -n scripts/run_attention_ablation.sh
```

CUDA/AMP tests skip on macOS and must be repeated on the remote GPU:

```bash
conda activate swin
python -m unittest losses.test_masked_hybrid_loss test_3d_baselines -v
```

For the complete workflow, see
[`docs/reproduction.md`](docs/reproduction.md) and
[`docs/dataset_versions.md`](docs/dataset_versions.md).
