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
        reference, _ = self.loss(logits.float(), target)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mixed_value, _ = self.loss(logits, target)
        self.assertTrue(torch.isfinite(mixed_value))
        self.assertTrue(torch.allclose(reference, mixed_value.float(), atol=3e-3, rtol=3e-3))
        mixed_value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
