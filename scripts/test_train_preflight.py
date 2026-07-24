"""Tests for launcher path, environment, and provenance preflight."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.train import configured_path, validate_runtime_inputs


def _config(data_root: Path, pretrained: str | None) -> dict:
    return {
        "experiment": {"task": "active_fire"},
        "data": {"preprocessed_root": str(data_root)},
        "model": {"pretrained_weights": pretrained},
        "training": {"sequence_length": 10, "interval": 3},
    }


class TrainPreflightTest(unittest.TestCase):
    def test_unresolved_environment_variable_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                configured_path("${MISSING_TRAINING_PATH}", "example.path")

    def test_runtime_inputs_are_verified_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset_train").mkdir()
            (root / "dataset_val").mkdir()
            required = (
                root / "dataset_train" / "af_train_img_seqtoseq_alll_10i_3.npy",
                root / "dataset_train" / "af_train_label_seqtoseq_alll_10i_3.npy",
                root / "dataset_val" / "af_val_img_seqtoseq_alll_10i_3.npy",
                root / "dataset_val" / "af_val_label_seqtoseq_alll_10i_3.npy",
            )
            for path in required:
                path.write_bytes(b"fixture")
            checkpoint = root / "swin.pth"
            checkpoint.write_bytes(b"verified checkpoint fixture")

            result = validate_runtime_inputs(_config(root, str(checkpoint)))

        self.assertEqual(result["pretrained"]["mode"], "checkpoint")
        self.assertEqual(
            result["pretrained"]["sha256"],
            hashlib.sha256(b"verified checkpoint fixture").hexdigest(),
        )
        self.assertEqual(set(result["dataset_files"]), {
            "train_images",
            "train_labels",
            "validation_images",
            "validation_labels",
        })

    def test_missing_pretrained_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset_train").mkdir()
            (root / "dataset_val").mkdir()
            for split, prefix in (("dataset_train", "train"), ("dataset_val", "val")):
                for kind in ("img", "label"):
                    (root / split / f"af_{prefix}_{kind}_seqtoseq_alll_10i_3.npy").write_bytes(
                        b"fixture"
                    )
            with self.assertRaises(FileNotFoundError):
                validate_runtime_inputs(_config(root, str(root / "missing.pth")))


if __name__ == "__main__":
    unittest.main()
