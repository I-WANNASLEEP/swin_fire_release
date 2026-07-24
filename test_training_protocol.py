"""Regression tests for checkpoint, early-stopping, and threshold boundaries."""

from __future__ import annotations

import unittest

import numpy as np

from training_protocol import (
    VALIDATION_THRESHOLDS,
    EarlyStopping,
    checkpoint_is_eligible,
    early_stopping_is_enabled,
    select_validation_threshold,
    validate_protocol,
)


class TrainingProtocolTest(unittest.TestCase):
    def test_epoch_boundaries_are_literal(self) -> None:
        self.assertFalse(checkpoint_is_eligible(50, 50))
        self.assertTrue(checkpoint_is_eligible(51, 50))
        self.assertFalse(early_stopping_is_enabled(100, 100))
        self.assertTrue(early_stopping_is_enabled(101, 100))

    def test_threshold_grid_is_exact_and_locked(self) -> None:
        self.assertEqual(
            VALIDATION_THRESHOLDS,
            (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
        )
        protocol = validate_protocol(
            checkpoint_after_epoch=50,
            early_stopping_after_epoch=100,
            thresholds=VALIDATION_THRESHOLDS,
        )
        self.assertEqual(protocol["first_checkpoint_epoch"], 51)
        self.assertEqual(protocol["first_early_stopping_epoch"], 101)
        with self.assertRaises(ValueError):
            validate_protocol(
                checkpoint_after_epoch=50,
                early_stopping_after_epoch=100,
                thresholds=(0.30, *VALIDATION_THRESHOLDS),
            )

    def test_threshold_selection_uses_only_locked_grid_and_positive_labels(self) -> None:
        probabilities = np.array([0.90, 0.58, 0.52, 0.38, 0.99])
        targets = np.array([255, 1, 0, 0, -1])
        result = select_validation_threshold(probabilities, targets)
        self.assertEqual(result["threshold"], 0.55)
        self.assertAlmostEqual(result["f1"], 1.0, places=5)
        self.assertEqual(result["true_positive"], 2)
        self.assertEqual(result["false_positive"], 0)

    def test_warmup_observes_best_f1_without_counting_patience(self) -> None:
        stopping = EarlyStopping(patience=2)
        self.assertFalse(stopping.update(0.80, enabled=False))
        self.assertFalse(stopping.update(0.70, enabled=False))
        self.assertEqual(stopping.best_f1, 0.80)
        self.assertEqual(stopping.counter, 0)
        self.assertFalse(stopping.update(0.70, enabled=True))
        self.assertEqual(stopping.counter, 1)
        self.assertTrue(stopping.update(0.70, enabled=True))
        self.assertEqual(stopping.counter, 2)


if __name__ == "__main__":
    unittest.main()
