"""Loss functions used by reproducible wildfire-segmentation experiments."""

from .masked_hybrid_loss import MaskedCrossEntropyLoss, MaskedHybridLoss

__all__ = ["MaskedCrossEntropyLoss", "MaskedHybridLoss"]
