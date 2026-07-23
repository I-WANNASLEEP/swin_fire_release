"""Airborne-manifest schema checks for cross-sensor validation."""

from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "event_id", "sensor_id", "acquisition_time", "source_path", "label_path",
    "channel_adaptation", "calibration_reference", "split",
}


def load_airborne_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Airborne manifest misses required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Airborne manifest contains no records.")
    return rows
