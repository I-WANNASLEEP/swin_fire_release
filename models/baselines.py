"""Compatibility imports for architecture baselines."""

from spatial_models.attentionunet import AttentionUnet
from spatial_models.unet import UNet
from spatial_models.unetr.unetr import UNETR
from spatial_models.swinunetr.swinunetr import SwinUNETR

__all__ = ["AttentionUnet", "UNet", "UNETR", "SwinUNETR"]
