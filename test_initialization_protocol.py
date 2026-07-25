"""Regression tests for MONAI-safe module-native initialization."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from initialization_protocol import audit_model_initialization


class TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(8, 2, 1),
        )


class InitializationProtocolTest(unittest.TestCase):
    def test_initialization_ablation_config_forbids_blanket_override(self) -> None:
        path = Path(__file__).resolve().parent / "configs" / "initialization_ablation.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        variants = config["variants"]
        self.assertEqual(
            {variant["strategy"] for variant in variants},
            {
                "module_native_defaults",
                "module_native_defaults_then_smart_checkpoint_load",
            },
        )
        self.assertTrue(
            all(
                variant["blanket_parameter_override"] is False
                for variant in variants
            )
        )

    def test_module_native_initialization_is_preserved_and_trainable(self) -> None:
        torch.manual_seed(42)
        model = TinySegmentationModel()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        rng_before = torch.random.get_rng_state().clone()

        report = audit_model_initialization(
            model,
            strategy="module_native_defaults",
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["segmentation_head"]["checked"])
        self.assertGreater(
            report["segmentation_head"]["minimum_critical_gradient_norm"],
            0.0,
        )
        self.assertTrue(
            torch.equal(rng_before, torch.random.get_rng_state()),
            "The audit must not consume the training RNG.",
        )
        for name, parameter in model.named_parameters():
            self.assertTrue(
                torch.equal(before[name], parameter.detach()),
                f"The audit modified {name}.",
            )
        self.assertTrue(torch.equal(model.seg_head[1].weight, torch.ones(8)))

    def test_historical_blanket_override_is_rejected(self) -> None:
        model = TinySegmentationModel()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(mean=0.0, std=0.02)
                elif parameter.ndim == 1:
                    parameter.zero_()
                else:
                    parameter.uniform_(-0.02, 0.02)

        with self.assertRaisesRegex(
            RuntimeError,
            "Zero-valued affine normalization scales",
        ):
            audit_model_initialization(
                model,
                strategy="historical_blanket_override",
            )

    def test_nonfinite_parameter_is_rejected(self) -> None:
        model = TinySegmentationModel()
        with torch.no_grad():
            model.seg_head[0].weight[0, 0, 0, 0] = float("nan")

        with self.assertRaisesRegex(RuntimeError, "Non-finite parameters"):
            audit_model_initialization(
                model,
                strategy="module_native_defaults",
            )

    def test_model_without_segmentation_head_is_still_audited(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        report = audit_model_initialization(
            model,
            strategy="module_native_defaults",
        )
        self.assertTrue(report["passed"])
        self.assertFalse(report["segmentation_head"]["checked"])


if __name__ == "__main__":
    unittest.main()
