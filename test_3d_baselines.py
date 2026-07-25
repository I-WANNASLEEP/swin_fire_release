"""Numerical regression tests for the two 3D architecture baselines."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from initialization_protocol import audit_model_initialization
from losses.masked_hybrid_loss import MaskedHybridLoss
from spatial_models.swinunetr.AutoregressiveAttention import (
    AutoregressiveAttention,
)
from spatial_models.swinunetr.WindowAttentionV2 import WindowAttentionV2
from spatial_models.swinunetr.swinunetr import SwinUNETR
from spatial_models.unet import UNet
from training_protocol import select_validation_threshold


class ThreeDimensionalBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(torch.get_num_threads(), 2))

    def _exercise_one_step(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
    ) -> None:
        target = torch.zeros(
            image.shape[0],
            1,
            *image.shape[2:],
            dtype=image.dtype,
        )
        target[..., 24:32, 24:32] = 1
        target[..., :2, :] = -1
        criterion = MaskedHybridLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-5)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits_before = model(image)
        self.assertEqual(
            tuple(logits_before.shape),
            (image.shape[0], 2, *image.shape[2:]),
        )
        self.assertTrue(torch.isfinite(logits_before).all())
        loss, _ = criterion(logits_before, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(torch.count_nonzero(gradient) > 0 for gradient in gradients))
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        self.assertGreater(float(gradient_norm), 0.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_after = model(image)
            probabilities = torch.softmax(logits_after, dim=1)[:, 1]
        self.assertTrue(torch.isfinite(logits_after).all())
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertGreater(float(probabilities.std()), 0.0)
        self.assertGreater(
            float((logits_after - logits_before.detach()).abs().mean()),
            0.0,
        )

        selected = select_validation_threshold(
            probabilities.numpy(),
            target[:, 0].numpy(),
        )
        for field in ("f1", "iou", "precision", "recall", "specificity"):
            self.assertTrue(np.isfinite(selected[field]))

    def test_singleton_temporal_window_buffers_are_finite(self) -> None:
        for attention_class in (WindowAttentionV2, AutoregressiveAttention):
            module = attention_class(
                dim=12,
                num_heads=3,
                window_size=(1, 4, 4),
            )
            self.assertTrue(torch.isfinite(module.relative_coords_table).all())
            self.assertTrue(
                torch.equal(
                    module.relative_coords_table[..., 0],
                    torch.zeros_like(module.relative_coords_table[..., 0]),
                )
            )

    def test_launcher_records_real_attention_and_five_seed_protocol(self) -> None:
        repository = Path(__file__).resolve().parent
        launcher = (
            repository / "scripts" / "run_architecture_baselines.sh"
        ).read_text(encoding="utf-8")
        config = (
            repository / "configs" / "architecture_baselines.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn('ARCH_BASELINE_SEEDS:-41 42 43 44 45', launcher)
        self.assertIn('model_attention="v2"', launcher)
        self.assertIn('-av "$model_attention"', launcher)
        self.assertIn("git diff --quiet", launcher)
        self.assertIn("continuing to the next baseline", launcher)
        self.assertIn("internal_attention: v2", config)

    def test_swinunetr3d_one_step_and_validation_are_finite(self) -> None:
        torch.manual_seed(42)
        model = SwinUNETR(
            image_size=(2, 64, 64),
            patch_size=(1, 2, 2),
            window_size=(1, 4, 4),
            in_channels=8,
            out_channels=2,
            depths=(1, 1, 1, 1),
            num_heads=(3, 6, 12, 24),
            feature_size=12,
            norm_name="batch",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            attn_version="v2",
            spatial_dims=3,
            use_checkpoint=False,
        )
        report = audit_model_initialization(
            model,
            strategy="module_native_defaults",
        )
        self.assertTrue(report["passed"])
        self.assertGreater(report["floating_buffer_tensors_checked"], 0)
        self._exercise_one_step(model, torch.randn(1, 8, 2, 64, 64))

    def test_unet3d_one_step_and_validation_are_finite(self) -> None:
        torch.manual_seed(42)
        model = UNet(
            spatial_dims=3,
            in_channels=8,
            out_channels=2,
            channels=(8, 16, 32, 64, 128),
            strides=(1, 2, 2),
        )
        report = audit_model_initialization(
            model,
            strategy="module_native_defaults",
        )
        self.assertTrue(report["passed"])
        self._exercise_one_step(model, torch.randn(1, 8, 2, 64, 64))


if __name__ == "__main__":
    unittest.main()
