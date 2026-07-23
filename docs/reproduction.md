# Complete Reproduction Guide

This document describes the exact steps to reproduce every experiment in the JEI
resubmission paper.

## Prerequisites

1. Conda environment: `ts-satfire-fixed` (Python 3.12.2, PyTorch 2.3.1+cu121)
   ```bash
   PYTHON=/home/congwei/miniconda3/envs/ts-satfire-fixed/bin/python
   ```

2. TS-SatFire dataset downloaded from Kaggle:
   https://www.kaggle.com/datasets/z789456sx/ts-satfire

3. Preprocessed NPY arrays at a known `DATA_ROOT` path, with structure:
   ```
   DATA_ROOT/
   ├── dataset_train/
   │   ├── af_train_img_seqtoseq_alll_10i_3.npy
   │   └── af_train_label_seqtoseq_alll_10i_3.npy
   └── dataset_val/
       ├── af_val_img_seqtoseq_alll_10i_3.npy
       └── af_val_label_seqtoseq_alll_10i_3.npy
   ```

4. Swin-Tiny pretrained weights from timm:
   https://github.com/rwightman/pytorch-image-models/releases

5. A sample manifest CSV with columns `event_id,split` tracking every
   generated window.

## Environment Setup

```bash
export PYTHON=/home/congwei/miniconda3/envs/ts-satfire-fixed/bin/python
export DATA_ROOT="/path/to/preprocessed_active_fire_arrays"
export PRETRAINED_PATH="/path/to/swin_tiny_patch4_window7_224.pth"
export SAMPLE_MANIFEST="/path/to/generated_sample_manifest.csv"
```

## Step 1: Validate Setup

```bash
# Verify splits
$PYTHON scripts/materialize_splits.py --check

# Verify loss correctness
$PYTHON -m unittest losses.test_masked_hybrid_loss

# Validate a single training config without launching
$PYTHON scripts/train.py --config configs/full_model.yaml --seed 41 --check
```

## Step 2: Tversky Parameter Selection (Validation Only)

Choose alpha/beta on the validation set. Do NOT look at test results during
this step.

```bash
bash scripts/run_tversky_grid_search.sh
```

This trains on the three candidates and reports validation-set metrics.

## Step 3: Run All Ablation Experiments

Execute each experiment independently. The scripts handle all seeds.

### Full Model (DCBAM, 5 seeds)
```bash
bash scripts/run_full_model.sh
```

### Attention Ablation (none, SE, CBAM, DCBAM; 5 seeds each)
```bash
bash scripts/run_attention_ablation.sh
```

### Progressive Ablation (Models A, B, C, D; 5 seeds each)
```bash
bash scripts/run_progressive_ablation.sh
```

### Architecture Baselines (SwinConvLSTM, SwinUNETR3D, UNet3D; 5 seeds each)
```bash
bash scripts/run_architecture_baselines.sh
```

### Initialization Ablation (pretrained vs random; 3 seeds each)
```bash
bash scripts/run_initialization_ablation.sh
```

## Step 4: Generate Paper Assets

After ALL training and evaluation is complete:

```bash
# Generate summary tables from raw per-event metrics
$PYTHON scripts/reproduce_all_tables.py \
    --input results/raw_metrics \
    --output results \
    --split test

# Generate convergence curves from epoch logs
$PYTHON scripts/reproduce_training_curves.py \
    --input results/training_runs \
    --output results
```

## Step 5: Verify Reproducibility

```bash
# Every table value must trace to a raw metric record
$PYTHON scripts/reproduce_all_tables.py \
    --input results/raw_metrics \
    --output results \
    --split test \
    --require-wandb \
    --wandb-mode online \
    --wandb-project swinfire_jei_resubmission_v2

# Check that all seeds have records
wc -l results/raw_metrics/*.json

# Verify min seeds requirement (3 for ablations, 5 for full model)
```

## Key Rules

- **Test set is NEVER used for threshold selection or hyperparameter tuning.**
- **Tversky alpha/beta is chosen on validation only.**
- **Optimal threshold is frozen from validation before test evaluation.**
- **Each seed is an independent training run.**
- **Epochs are NOT treated as independent experiments.**
- **Confidence intervals are bootstrapped over independent fire events.**
