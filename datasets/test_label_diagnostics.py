"""Tests for exact, chunked segmentation-label diagnostics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from datasets.label_diagnostics import summarize_label_file


class LabelDiagnosticsTest(unittest.TestCase):
    def test_counts_binary_255_and_ignored_encodings(self) -> None:
        labels = np.array(
            [
                [
                    [[0, 1], [255, -1]],
                    [[0, 0], [0, 0]],
                ]
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.npy"
            np.save(path, labels)
            result = summarize_label_file(path, label_index=0, chunk_samples=1)

        self.assertTrue(result["contract_valid"])
        self.assertEqual(result["total_pixels"], 4)
        self.assertEqual(result["valid_pixels"], 3)
        self.assertEqual(result["ignored_pixels"], 1)
        self.assertEqual(result["background_pixels"], 1)
        self.assertEqual(result["fire_pixels"], 2)
        self.assertEqual(result["nonunit_positive_pixels"], 1)
        self.assertEqual(result["unique_values"], [-1.0, 0.0, 1.0, 255.0])

    def test_flags_nonfinite_and_unexpected_negative_values(self) -> None:
        labels = np.array([[[[0, -2, np.nan, np.inf]]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npy"
            np.save(path, labels)
            result = summarize_label_file(path, label_index=0)

        self.assertFalse(result["contract_valid"])
        self.assertEqual(result["nonfinite_pixels"], 2)
        self.assertEqual(result["unexpected_negative_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
