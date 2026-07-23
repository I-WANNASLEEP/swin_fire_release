#!/usr/bin/env python3
"""Evaluate one frozen checkpoint on one declared event and emit a raw record.

Inputs are probability/logit and target arrays already aligned to the same event.
The script refuses undeclared events and ignores labels equal to -1 in every
metric. Threshold selection belongs to validation; test evaluation requires a
previously frozen threshold supplied on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.splits import load_event_ids, require_member  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fire_probability(prediction: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Accept probabilities [B,...] or two-class logits [B,2,...]."""
    if prediction.ndim == len(target_shape) + 1 and prediction.shape[1] == 2:
        shifted = prediction - prediction.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp[:, 1, ...] / exp.sum(axis=1)
    return prediction


def canonical_target(target: np.ndarray) -> np.ndarray:
    # Only remove a singleton *class* axis.  A valid one-step temporal tensor
    # such as [B, T=1, H, W] must remain unchanged.
    if target.ndim >= 5 and target.shape[1] == 1:
        return target[:, 0, ...]
    return target


def masked_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float | int]:
    if probability.shape != target.shape:
        raise ValueError(f"Prediction shape {probability.shape} does not match target shape {target.shape}.")
    valid = target != -1
    if not np.any(valid):
        raise ValueError("The evaluated event contains no valid target pixels.")
    prediction = probability[valid] >= threshold
    truth = target[valid] > 0
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    tn = int(np.sum(~prediction & ~truth))
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "valid_pixels": int(valid.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attention", default="not_applicable")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--sample-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_file = PROJECT_ROOT / "splits" / f"{args.split}_event_ids.txt"
    require_member(args.event_id, load_event_ids(split_file), args.split)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("The frozen threshold must be in [0, 1].")
    target = canonical_target(np.load(args.target))
    probability = fire_probability(np.load(args.prediction), target.shape)
    record = {
        "experiment": args.experiment,
        "model": args.model,
        "attention": args.attention,
        "seed": args.seed,
        "split": args.split,
        "event_id": args.event_id,
        "threshold": args.threshold,
        "prediction_sha256": sha256(args.prediction),
        "target_sha256": sha256(args.target),
        "checkpoint_sha256": args.checkpoint_sha256,
        "dataset_sha256": args.dataset_sha256,
        "sample_manifest_sha256": args.sample_manifest_sha256,
        **masked_metrics(probability, target, args.threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote raw metrics: {args.output}")


if __name__ == "__main__":
    main()
