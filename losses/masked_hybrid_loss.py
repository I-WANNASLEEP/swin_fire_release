"""Masked loss functions for two-class spatio-temporal fire segmentation.

The model emits two logits per pixel: background (class 0) and fire (class 1).
Pixels whose target equals ``ignore_index`` are excluded before every reduction,
so they contribute neither loss nor gradient.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedHybridLoss(nn.Module):
    """Masked Tversky + focal + cross-entropy loss for two-class logits.

    ``alpha`` weights false positives and ``beta`` weights false negatives.
    Select them on validation data only.  ``alpha=beta=0.5`` is symmetric and
    Dice-like; it is not a recall-specific configuration.
    """

    def __init__(
        self,
        tversky_weight: float = 0.4,
        focal_weight: float = 0.3,
        ce_weight: float = 0.3,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 3.0,
        ignore_index: int = -1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(tversky_weight, focal_weight, ce_weight) < 0:
            raise ValueError("Loss weights must be non-negative.")
        if alpha < 0 or beta < 0:
            raise ValueError("Tversky alpha and beta must be non-negative.")
        if gamma < 0:
            raise ValueError("Focal gamma must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.tversky_weight = float(tversky_weight)
        self.focal_weight = float(focal_weight)
        self.ce_weight = float(ce_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    @staticmethod
    def fire_probability(logits: torch.Tensor) -> torch.Tensor:
        """Return class-1 probabilities from two-class logits using Softmax."""
        if logits.ndim < 3 or logits.shape[1] != 2:
            raise ValueError("Expected logits with shape [B, 2, ...].")
        return torch.softmax(logits, dim=1)[:, 1, ...]

    def _prepare_target(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:
            if target.shape[1] != 1:
                raise ValueError("Targets with a class dimension must be [B, 1, ...].")
            target = target[:, 0, ...]
        expected_shape = (logits.shape[0], *logits.shape[2:])
        if target.shape != expected_shape:
            raise ValueError(
                f"Target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}."
            )
        return target.long()

    def forward(
        self, logits: torch.Tensor | Sequence[torch.Tensor], target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all three components over valid pixels only."""
        if isinstance(logits, (list, tuple)):
            if not logits:
                raise ValueError("The logits sequence is empty.")
            logits = logits[0]
        if logits.ndim < 3 or logits.shape[1] != 2:
            raise ValueError("MaskedHybridLoss requires two-class logits [B, 2, ...].")

        target = self._prepare_target(logits, target)
        valid_mask = target.ne(self.ignore_index)
        zero = logits.sum() * 0.0
        if not torch.any(valid_mask):
            return zero, {
                "tversky_loss": 0.0,
                "focal_loss": 0.0,
                "ce_loss": 0.0,
                "valid_pixels": 0.0,
            }

        log_probabilities = F.log_softmax(logits, dim=1)
        log_p_background = log_probabilities[:, 0, ...]
        log_p_fire = log_probabilities[:, 1, ...]
        p_fire = log_p_fire.exp()
        target_valid = target[valid_mask]
        fire_target = target_valid.to(dtype=logits.dtype)
        p_fire_valid = p_fire[valid_mask]

        true_positive = (p_fire_valid * fire_target).sum()
        false_positive = (p_fire_valid * (1.0 - fire_target)).sum()
        false_negative = ((1.0 - p_fire_valid) * fire_target).sum()
        tversky = (true_positive + self.eps) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.eps
        )
        tversky_loss = 1.0 - tversky

        log_p_t = torch.where(
            target_valid.eq(1),
            log_p_fire[valid_mask],
            log_p_background[valid_mask],
        )
        focal_loss = (-torch.pow(1.0 - log_p_t.exp(), self.gamma) * log_p_t).mean()

        safe_target = target.clamp(min=0, max=1)
        ce_map = -log_probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        ce_loss = ce_map[valid_mask].mean()
        total_loss = (
            self.tversky_weight * tversky_loss
            + self.focal_weight * focal_loss
            + self.ce_weight * ce_loss
        )
        return total_loss, {
            "tversky_loss": float(tversky_loss.detach().cpu()),
            "focal_loss": float(focal_loss.detach().cpu()),
            "ce_loss": float(ce_loss.detach().cpu()),
            "valid_pixels": float(valid_mask.sum().detach().cpu()),
        }


class MaskedCrossEntropyLoss(nn.Module):
    """Correct CE-only comparator with identical ignored-pixel semantics."""

    def __init__(self, ignore_index: int = -1) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(
        self, logits: torch.Tensor | Sequence[torch.Tensor], target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(logits, (list, tuple)):
            if not logits:
                raise ValueError("The logits sequence is empty.")
            logits = logits[0]
        if logits.ndim < 3 or logits.shape[1] != 2:
            raise ValueError("MaskedCrossEntropyLoss requires two-class logits [B, 2, ...].")
        target = MaskedHybridLoss(ignore_index=self.ignore_index)._prepare_target(logits, target)
        valid_mask = target.ne(self.ignore_index)
        if not torch.any(valid_mask):
            zero = logits.sum() * 0.0
            return zero, {"tversky_loss": 0.0, "focal_loss": 0.0, "ce_loss": 0.0, "valid_pixels": 0.0}
        log_probabilities = F.log_softmax(logits, dim=1)
        safe_target = target.clamp(min=0, max=1)
        ce_map = -log_probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        ce_loss = ce_map[valid_mask].mean()
        return ce_loss, {
            "tversky_loss": 0.0,
            "focal_loss": 0.0,
            "ce_loss": float(ce_loss.detach().cpu()),
            "valid_pixels": float(valid_mask.sum().detach().cpu()),
        }
