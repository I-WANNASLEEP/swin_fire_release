# Airborne Cross-Sensor Validation Protocol

## Status

The airborne cross-sensor validation experiment is **defined but not yet
executed**. No airborne metric is claimed in the current manuscript revision
until the complete protocol below is followed.

## Requirements

1. **Independent held-out events**: Airborne events must not overlap with any
   training, validation, or test satellite events.

2. **Explicit channel adaptation**: Zero-filling missing satellite channels is
   NOT an acceptable cross-sensor adaptation method. Each airborne record must
   specify the channel adaptation method used (e.g., spectral band matching,
   calibration-based mapping, domain adaptation model).

3. **Calibration reference**: Every airborne record must cite a calibration
   reference (satellite overpass, ground truth, or sensor specification).

4. **Test-only evaluation**: Airborne data is exclusively for evaluation. No
   airborne event may be used for threshold selection, Tversky parameter
   tuning, or any other hyperparameter optimization.

## Manifest Format

Each airborne sample must be documented in a CSV manifest with these columns:

| Column                | Description                                    |
|-----------------------|------------------------------------------------|
| event_id              | Unique fire event identifier                   |
| sensor_id             | Airborne sensor/platform ID                    |
| acquisition_time      | ISO 8601 timestamp of acquisition              |
| source_path           | Path to the preprocessed image array           |
| label_path            | Path to the corresponding ground-truth label   |
| channel_adaptation    | Method used to match 8-channel satellite input |
| calibration_reference | Citation for the calibration data used         |
| split                 | Must be "test" for all rows                    |

## Validation Script

```bash
$PYTHON scripts/evaluate_airborne.py \
    --manifest path/to/airborne_manifest.csv \
    --output results/raw_metrics/airborne_validation.json
```

This script:
- Rejects any non-test split
- Rejects zero-fill channel adaptation
- Requires calibration references
- Does NOT evaluate any model; it only validates the protocol

## Metric Reporting

Once the protocol is validated AND predictions are generated with a frozen
checkpoint:

1. Generate per-event metric records using `scripts/evaluate.py` with the
   airborne events (split must be in the test manifest)
2. Report metrics separately as a cross-sensor validation experiment
3. Do not aggregate airborne and satellite metrics into a single table
