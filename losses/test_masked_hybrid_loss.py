"""Regression tests for the corrected masked two-class Hybrid Loss.

Run CPU tests with ``python -m unittest losses.test_masked_hybrid_loss``.
CUDA and CUDA-AMP checks are skipped automatically when CUDA is unavailable.
"""

from __future__ import annotations

import unittest

import torch

from losses.masked_hybrid_loss import MaskedCrossEntropyLoss, MaskedHybridLoss


class MaskedHybridLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.loss = MaskedHybridLoss()

    @staticmethod
    def _logits(device: str = "cpu", requires_grad: bool = False) -> torch.Tensor:
        return torch.randn(2, 2, 2, 3, 4, device=device, requires_grad=requires_grad)

    @staticmethod
    def _target(device: str = "cpu") -> torch.Tensor:
        target = torch.randint(0, 2, (2, 1, 2, 3, 4), device=device)
        target[..., 0, 0] = -1
        target[..., 2, 3] = -1
        return target

    def test_invalid_predictions_do_not_change_loss(self) -> None:
        logits = self._logits()
        target = self._target()
        changed = logits.clone()
        invalid = target[:, 0].eq(-1)
        changed[:, 0][invalid] = 1_000.0
        changed[:, 1][invalid] = -1_000.0
        expected, _ = self.loss(logits, target)
        observed, _ = self.loss(changed, target)
        self.assertTrue(torch.allclose(expected, observed, atol=1e-6, rtol=1e-6))

    def test_invalid_gradients_are_exactly_zero(self) -> None:
        logits = self._logits(requires_grad=True)
        target = self._target()
        value, _ = self.loss(logits, target)
        value.backward()
        invalid = target[:, 0].eq(-1).unsqueeze(1).expand_as(logits)
        self.assertTrue(torch.equal(logits.grad[invalid], torch.zeros_like(logits.grad[invalid])))

    def test_all_invalid_is_finite_and_differentiable_zero(self) -> None:
        logits = self._logits(requires_grad=True)
        target = torch.full((2, 1, 2, 3, 4), -1, dtype=torch.long)
        value, components = self.loss(logits, target)
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(components["valid_pixels"], 0.0)
        value.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_background_fire_and_rare_fire_backward(self) -> None:
        background = torch.zeros((2, 1, 2, 3, 4), dtype=torch.long)
        fire = torch.ones((2, 1, 2, 3, 4), dtype=torch.long)
        rare_fire = background.clone()
        rare_fire[..., 1, 2, 3] = 1
        for target in (background, fire, rare_fire):
            logits = self._logits(requires_grad=True)
            value, _ = self.loss(logits, target)
            self.assertTrue(torch.isfinite(value))
            value.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_positive_label_encodings_have_identical_loss_and_gradients(self) -> None:
        binary = self._target()
        encoded = binary.clone()
        encoded[encoded.eq(1)] = 255
        logits_binary = self._logits(requires_grad=True)
        logits_encoded = logits_binary.detach().clone().requires_grad_(True)

        binary_value, binary_components = self.loss(logits_binary, binary)
        encoded_value, encoded_components = self.loss(logits_encoded, encoded)
        binary_value.backward()
        encoded_value.backward()

        self.assertTrue(torch.allclose(binary_value, encoded_value, atol=1e-7, rtol=1e-7))
        for key in ("tversky_loss", "focal_loss", "ce_loss", "valid_pixels"):
            self.assertAlmostEqual(binary_components[key], encoded_components[key], places=7)
        self.assertTrue(
            torch.allclose(logits_binary.grad, logits_encoded.grad, atol=1e-7, rtol=1e-7)
        )

    def test_supported_targets_produce_nonnegative_components(self) -> None:
        targets = []
        for positive_value in (1, 255):
            target = torch.zeros((1, 1, 1, 1, 10_000), dtype=torch.long)
            target[..., 0] = positive_value
            targets.append(target)
        logits_template = torch.empty((1, 2, 1, 1, 10_000))
        logits_template[:, 0] = 10.0
        logits_template[:, 1] = -10.0
        logits_template[:, 0, ..., 0] = -1.0
        logits_template[:, 1, ..., 0] = 1.0

        for target in targets:
            logits = logits_template.clone().requires_grad_(True)
            value, components = self.loss(logits, target)
            self.assertGreaterEqual(float(value.detach()), 0.0)
            for key in ("tversky_loss", "focal_loss", "ce_loss"):
                self.assertGreaterEqual(components[key], 0.0)
            value.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rejects_nan_infinite_and_unexpected_negative_targets(self) -> None:
        logits = self._logits()
        bad_targets = []
        for value in (float("nan"), float("inf"), -2.0):
            target = self._target().float()
            target[:, :, 0, 1, 1] = value
            bad_targets.append(target)
        for target in bad_targets:
            with self.assertRaises(ValueError):
                self.loss(logits, target)

    def test_softmax_probability_sums_to_one(self) -> None:
        logits = self._logits()
        fire = self.loss.fire_probability(logits)
        background = torch.softmax(logits, dim=1)[:, 0]
        self.assertTrue(torch.allclose(fire + background, torch.ones_like(fire), atol=1e-6))

    def test_ce_only_has_the_same_invalid_pixel_semantics(self) -> None:
        criterion = MaskedCrossEntropyLoss()
        logits = self._logits(requires_grad=True)
        target = self._target()
        changed = logits.detach().clone().requires_grad_(True)
        invalid = target[:, 0].eq(-1)
        changed.data[:, 0][invalid] = 1_000.0
        changed.data[:, 1][invalid] = -1_000.0
        expected, _ = criterion(logits, target)
        observed, _ = criterion(changed, target)
        self.assertTrue(torch.allclose(expected, observed, atol=1e-6, rtol=1e-6))
        expected.backward()
        mask = invalid.unsqueeze(1).expand_as(logits)
        self.assertTrue(torch.equal(logits.grad[mask], torch.zeros_like(logits.grad[mask])))

    def test_ce_only_binarizes_all_positive_label_encodings(self) -> None:
        criterion = MaskedCrossEntropyLoss()
        binary = self._target()
        encoded = binary.clone()
        encoded[encoded.eq(1)] = 255
        logits_binary = self._logits(requires_grad=True)
        logits_encoded = logits_binary.detach().clone().requires_grad_(True)
        binary_value, _ = criterion(logits_binary, binary)
        encoded_value, _ = criterion(logits_encoded, encoded)
        binary_value.backward()
        encoded_value.backward()
        self.assertTrue(torch.allclose(binary_value, encoded_value, atol=1e-7, rtol=1e-7))
        self.assertTrue(
            torch.allclose(logits_binary.grad, logits_encoded.grad, atol=1e-7, rtol=1e-7)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_matches_cpu(self) -> None:
        logits_cpu = self._logits()
        target_cpu = self._target()
        cpu_value, _ = self.loss(logits_cpu, target_cpu)
        cuda_value, _ = self.loss(logits_cpu.cuda(), target_cpu.cuda())
        self.assertTrue(torch.allclose(cpu_value, cuda_value.cpu(), atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_amp_is_finite_and_stable(self) -> None:
        logits = self._logits(device="cuda", requires_grad=True)
        target = self._target(device="cuda")
        target[target.eq(1)] = 255
        reference, _ = self.loss(logits.float(), target)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mixed_value, _ = self.loss(logits, target)
        self.assertTrue(torch.isfinite(mixed_value))
        self.assertTrue(torch.allclose(reference, mixed_value.float(), atol=3e-3, rtol=3e-3))
        mixed_value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
