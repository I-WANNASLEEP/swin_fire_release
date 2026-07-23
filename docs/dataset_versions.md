# Dataset Version Lock

## Source

- **Name**: TS-SatFire
- **URL**: https://www.kaggle.com/datasets/z789456sx/ts-satfire
- **DOI**: https://doi.org/10.1038/s41597-025-06271-3
- **Upstream code**: https://github.com/zhaoyutim/TS-SatFire
- **Pinned commit**: da7573219967edfe17e317310b66fd8a913f4a2e
- **Pinned on**: 2026-07-23
- **Task**: Active fire segmentation (8 channels)

## Preprocessing

All upstream preprocessing procedures are used without modification. The
official event-level training (125 events), validation (13 events), and test
(17 events) partition released with TS-SatFire is followed exactly.

No data is redistributed in this repository. Before training, you must:

1. Download the TS-SatFire dataset from Kaggle
2. Run the upstream preprocessing code at the pinned commit
3. Generate the aggregate NPY arrays from the official event lists
4. Record the archive SHA256 and download date in your experiment manifest

## Channel Order

| Index | Channel  | Description            |
|-------|----------|------------------------|
| 0     | I1_day   | Visible blue           |
| 1     | I2_day   | Visible green          |
| 2     | I3_day   | Visible red            |
| 3     | I4_day   | Near-infrared          |
| 4     | I5_day   | Short-wave infrared    |
| 5     | M11_day  | Thermal infrared       |
| 6     | I4_night | Near-infrared (night)  |
| 7     | I5_night | Short-wave infrared (night) |

## Normalization

Z-score normalization per channel using statistics computed from the
training partition. See `metadata/normalization_statistics.json`.

## Event Partition

| Split      | Events | Source                      |
|------------|--------|-----------------------------|
| Training   | 125    | ROI CSVs 2017-2020 minus val|
| Validation | 13     | Fixed upstream list         |
| Test       | 17     | Named active-fire events    |

See `splits/train_event_ids.txt`, `splits/validation_event_ids.txt`,
`splits/test_event_ids.txt`.

## Time Windows

- Sequence length: 10 time steps
- Interval: 3 hours
- No overlap between train/val/test events
- Test events are held out from ALL tuning (threshold, Tversky parameters)
