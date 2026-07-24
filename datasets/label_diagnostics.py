"""Auditable label-domain diagnostics for preprocessed segmentation arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def summarize_label_file(
    path: str | Path,
    label_index: int,
    *,
    ignore_index: int = -1,
    chunk_samples: int = 1,
    max_unique_values: int = 32,
) -> dict[str, Any]:
    """Scan one label channel and return exact domain counts.

    The scan is chunked over the sample dimension so large memory-mapped arrays
    do not need to be loaded at once.  Counts/min/max are exact.  The displayed
    unique-value list is bounded and declares when it is truncated.
    """
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")
    if max_unique_values <= 0:
        raise ValueError("max_unique_values must be positive.")

    label_path = Path(path).expanduser().resolve()
    labels = np.load(label_path, mmap_mode="r")
    if labels.ndim < 3:
        raise ValueError(f"Expected labels shaped [N, C, ...], got {labels.shape}.")
    if not 0 <= label_index < labels.shape[1]:
        raise ValueError(
            f"label_index {label_index} is outside channel dimension {labels.shape[1]}."
        )

    counts = {
        "total_pixels": 0,
        "finite_pixels": 0,
        "nonfinite_pixels": 0,
        "ignored_pixels": 0,
        "valid_pixels": 0,
        "background_pixels": 0,
        "fire_pixels": 0,
        "nonunit_positive_pixels": 0,
        "unexpected_negative_pixels": 0,
    }
    minimum = None
    maximum = None
    unique_values: set[float] = set()
    unique_truncated = False

    for start in range(0, labels.shape[0], chunk_samples):
        chunk = np.asarray(labels[start : start + chunk_samples, label_index])
        finite = np.isfinite(chunk)
        ignored = finite & (chunk == ignore_index)
        valid = finite & ~ignored
        background = valid & (chunk == 0)
        fire = valid & (chunk > 0)
        nonunit_positive = fire & (chunk != 1)
        unexpected_negative = valid & (chunk < 0)

        counts["total_pixels"] += int(chunk.size)
        counts["finite_pixels"] += int(finite.sum())
        counts["nonfinite_pixels"] += int((~finite).sum())
        counts["ignored_pixels"] += int(ignored.sum())
        counts["valid_pixels"] += int(valid.sum())
        counts["background_pixels"] += int(background.sum())
        counts["fire_pixels"] += int(fire.sum())
        counts["nonunit_positive_pixels"] += int(nonunit_positive.sum())
        counts["unexpected_negative_pixels"] += int(unexpected_negative.sum())

        if np.any(valid):
            valid_values = chunk[valid]
            chunk_min = float(valid_values.min())
            chunk_max = float(valid_values.max())
            minimum = chunk_min if minimum is None else min(minimum, chunk_min)
            maximum = chunk_max if maximum is None else max(maximum, chunk_max)

        if not unique_truncated and np.any(finite):
            for value in np.unique(chunk[finite]):
                unique_values.add(float(value))
                if len(unique_values) > max_unique_values:
                    unique_truncated = True
                    unique_values = set(sorted(unique_values)[:max_unique_values])
                    break

    contract_valid = (
        counts["nonfinite_pixels"] == 0
        and counts["unexpected_negative_pixels"] == 0
    )
    return {
        "path": str(label_path),
        "shape": [int(value) for value in labels.shape],
        "dtype": str(labels.dtype),
        "label_index": int(label_index),
        "ignore_index": int(ignore_index),
        "positive_label_rule": "target > 0",
        "valid_min": minimum,
        "valid_max": maximum,
        "unique_values": sorted(unique_values),
        "unique_values_truncated": unique_truncated,
        "contract_valid": contract_valid,
        **counts,
    }
