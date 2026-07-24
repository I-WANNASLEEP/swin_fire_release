"""Deterministic validation-selection and stopping protocol for training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


VALIDATION_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def validate_protocol(
    *,
    checkpoint_after_epoch: int,
    early_stopping_after_epoch: int,
    thresholds: Iterable[float],
) -> dict[str, object]:
    """Validate and normalize the user-locked training protocol."""
    threshold_tuple = tuple(float(value) for value in thresholds)
    if threshold_tuple != VALIDATION_THRESHOLDS:
        raise ValueError(
            "validation thresholds are locked to "
            f"{list(VALIDATION_THRESHOLDS)}, got {list(threshold_tuple)}."
        )
    if checkpoint_after_epoch < 0:
        raise ValueError("checkpoint_after_epoch must be non-negative.")
    if early_stopping_after_epoch < checkpoint_after_epoch:
        raise ValueError(
            "early_stopping_after_epoch cannot precede checkpoint_after_epoch."
        )
    return {
        "checkpoint_after_epoch": int(checkpoint_after_epoch),
        "first_checkpoint_epoch": int(checkpoint_after_epoch) + 1,
        "early_stopping_after_epoch": int(early_stopping_after_epoch),
        "first_early_stopping_epoch": int(early_stopping_after_epoch) + 1,
        "validation_thresholds": list(threshold_tuple),
        "threshold_comparator": "probability >= threshold",
        "checkpoint_metric": "validation_f1_at_selected_threshold",
    }


def checkpoint_is_eligible(epoch_number: int, checkpoint_after_epoch: int) -> bool:
    """Return True only after the configured checkpoint warm-up is complete."""
    return int(epoch_number) > int(checkpoint_after_epoch)


def early_stopping_is_enabled(
    epoch_number: int, early_stopping_after_epoch: int
) -> bool:
    """Return True only after the configured no-early-stop period."""
    return int(epoch_number) > int(early_stopping_after_epoch)


@dataclass
class EarlyStopping:
    """Track validation F1 throughout warm-up but count patience only when enabled."""

    patience: int
    min_delta: float = 0.0001
    best_f1: float | None = None
    counter: int = 0
    early_stop: bool = False
    last_improved: bool = False

    def update(self, validation_f1: float, *, enabled: bool) -> bool:
        value = float(validation_f1)
        if not np.isfinite(value):
            raise ValueError("validation_f1 must be finite.")

        improved = self.best_f1 is None or value > self.best_f1 + self.min_delta
        self.last_improved = improved
        if improved:
            self.best_f1 = value
            self.counter = 0
        elif enabled:
            self.counter += 1
        else:
            self.counter = 0

        self.early_stop = bool(enabled and self.counter >= self.patience)
        return self.early_stop


def select_validation_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: Iterable[float] = VALIDATION_THRESHOLDS,
    ignore_index: int = -1,
    eps: float = 1e-6,
) -> dict[str, object]:
    """Traverse only the locked thresholds and return the highest validation F1."""
    threshold_tuple = tuple(float(value) for value in thresholds)
    if threshold_tuple != VALIDATION_THRESHOLDS:
        raise ValueError(
            "Threshold selection must use the locked validation threshold grid."
        )

    probability_array = np.asarray(probabilities).reshape(-1)
    target_array = np.asarray(targets).reshape(-1)
    if probability_array.shape != target_array.shape:
        raise ValueError("Probability and target arrays must have identical shapes.")
    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities contain NaN or infinite values.")
    if not np.isfinite(target_array).all():
        raise ValueError("Targets contain NaN or infinite values.")

    valid = target_array != ignore_index
    valid_targets = target_array[valid]
    if valid_targets.size and np.any(valid_targets < 0):
        raise ValueError("Valid targets must be 0 or positive fire values.")
    valid_probabilities = probability_array[valid]
    binary_targets = valid_targets > 0

    best: dict[str, object] | None = None
    for threshold in threshold_tuple:
        predictions = valid_probabilities >= threshold
        true_positive = int(np.sum(predictions & binary_targets))
        false_positive = int(np.sum(predictions & ~binary_targets))
        false_negative = int(np.sum(~predictions & binary_targets))
        true_negative = int(np.sum(~predictions & ~binary_targets))
        precision = true_positive / (true_positive + false_positive + eps)
        recall = true_positive / (true_positive + false_negative + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = true_positive / (
            true_positive + false_positive + false_negative + eps
        )
        specificity = true_negative / (true_negative + false_positive + eps)
        candidate = {
            "threshold": float(threshold),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "iou": float(iou),
            "specificity": float(specificity),
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "confusion_matrix": np.array(
                [[true_negative, false_positive], [false_negative, true_positive]]
            ),
        }
        # Strict comparison means an exact tie keeps the first traversed
        # threshold, making the ascending-grid behavior deterministic.
        if best is None or candidate["f1"] > best["f1"]:
            best = candidate

    if best is None:
        raise RuntimeError("The locked validation threshold grid is empty.")
    return best
