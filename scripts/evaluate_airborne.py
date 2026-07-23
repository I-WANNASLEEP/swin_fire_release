#!/usr/bin/env python3
"""Validate the cross-sensor airborne protocol before any result is reported.

This intentionally rejects the legacy practice of silently zero-filling six
satellite channels. A cross-sensor result requires an explicit channel-adaptation
method, calibration reference, independent event IDs, and held-out labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REQUIRED_COLUMNS = {
    "event_id", "sensor_id", "acquisition_time", "source_path", "label_path",
    "channel_adaptation", "calibration_reference", "split",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Airborne manifest misses required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Airborne manifest has no samples.")
    event_ids = {row["event_id"] for row in rows}
    invalid = [row["event_id"] for row in rows if row["split"] != "test"]
    zero_fill = [row["event_id"] for row in rows if "zero" in row["channel_adaptation"].lower()]
    if invalid:
        raise ValueError("Airborne validation must contain held-out test events only.")
    if zero_fill:
        raise ValueError("Zero-filled missing channels are not an acceptable cross-sensor adaptation.")
    if any(not row["calibration_reference"].strip() for row in rows):
        raise ValueError("Every airborne record needs a calibration reference.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "n_samples": len(rows), "n_independent_events": len(event_ids),
        "status": "protocol_validated_not_evaluated",
        "next_step": "Run frozen-checkpoint predictions and emit per-event masked metrics without test-set tuning.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated airborne protocol for {len(event_ids)} held-out event(s).")


if __name__ == "__main__":
    main()
