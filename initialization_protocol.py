"""Safe initialization auditing for PyTorch and MONAI-backed models.

Random initialization means retaining each module's constructor-provided
initialization.  A blanket pass over ``model.parameters()`` is unsafe because
architectures such as MONAI SwinUNETR mix linear/attention parameters with
normalization scales that require different initialization semantics.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_NORMALIZATION_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
)


def _gradient_norm(gradient: torch.Tensor | None) -> float:
    if gradient is None:
        return 0.0
    return float(gradient.detach().float().norm().cpu())


def _audit_segmentation_head(model: nn.Module) -> dict[str, Any]:
    """Verify that a small synthetic loss reaches the spatial prediction path.

    ``torch.autograd.grad`` is used instead of ``backward`` so the check does
    not populate parameter ``.grad`` fields.  The deterministic linspace probe
    does not advance any global RNG state.
    """

    head = getattr(model, "seg_head", None)
    if not isinstance(head, nn.Module):
        return {
            "checked": False,
            "reason": "model_has_no_seg_head",
        }

    convolutions = [
        (name, module)
        for name, module in head.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if len(convolutions) < 2:
        raise RuntimeError(
            "Initialization audit requires at least two Conv2d layers in "
            "model.seg_head."
        )

    first_name, first_conv = convolutions[0]
    final_name, final_conv = convolutions[-1]
    if first_conv.weight.device.type != "cpu":
        raise RuntimeError(
            "Run the initialization audit before model.to(device) so the "
            "side-effect-free probe remains a lightweight CPU check."
        )
    if final_conv.out_channels < 2:
        raise RuntimeError(
            "Segmentation-head initialization audit requires at least two "
            "output classes."
        )

    critical_parameters: list[tuple[str, nn.Parameter]] = [
        (f"seg_head.{first_name}.weight", first_conv.weight),
        (f"seg_head.{final_name}.weight", final_conv.weight),
    ]
    for name, module in head.named_modules():
        if (
            isinstance(module, _NORMALIZATION_TYPES)
            and module.weight is not None
            and module.weight.requires_grad
        ):
            critical_parameters.append(
                (f"seg_head.{name}.weight", module.weight)
            )

    probe_values = torch.linspace(
        -1.0,
        1.0,
        steps=2 * first_conv.in_channels * 8 * 8,
        dtype=first_conv.weight.dtype,
        device=first_conv.weight.device,
    )
    probe = probe_values.reshape(2, first_conv.in_channels, 8, 8)

    was_training = head.training
    try:
        head.eval()
        logits = head(probe)
        if logits.ndim != 4 or logits.shape[1] != final_conv.out_channels:
            raise RuntimeError(
                "Segmentation head produced an unexpected shape during the "
                f"initialization audit: {tuple(logits.shape)}."
            )
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(
                "Segmentation head produced non-finite logits before training."
            )

        target = (
            torch.arange(
                logits.shape[0] * logits.shape[2] * logits.shape[3],
                device=logits.device,
            )
            .reshape(logits.shape[0], logits.shape[2], logits.shape[3])
            .remainder(final_conv.out_channels)
        )
        loss = F.cross_entropy(logits.float(), target)
        gradients = torch.autograd.grad(
            loss,
            [parameter for _, parameter in critical_parameters],
            allow_unused=True,
        )
    finally:
        head.train(was_training)

    gradient_norms = {
        name: _gradient_norm(gradient)
        for (name, _), gradient in zip(critical_parameters, gradients)
    }
    blocked = [
        name
        for name, value in gradient_norms.items()
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0
    ]
    if blocked:
        raise RuntimeError(
            "Initialization blocks gradient flow through critical segmentation "
            "head parameters: "
            + ", ".join(blocked)
        )

    return {
        "checked": True,
        "probe_shape": list(probe.shape),
        "logit_std": float(logits.detach().float().std().cpu()),
        "loss": float(loss.detach().cpu()),
        "critical_gradient_norms": gradient_norms,
        "minimum_critical_gradient_norm": min(gradient_norms.values()),
    }


def audit_model_initialization(
    model: nn.Module,
    *,
    strategy: str,
) -> dict[str, Any]:
    """Fail fast when initialization is non-finite or blocks normalization.

    The function is intentionally read-only with respect to parameters and RNG
    state.  It supports MONAI models without importing MONAI or assuming a
    particular MONAI version.
    """

    parameter_tensors = 0
    parameter_values = 0
    nonfinite_parameters: list[str] = []
    for name, parameter in model.named_parameters():
        parameter_tensors += 1
        parameter_values += parameter.numel()
        if not bool(torch.isfinite(parameter.detach()).all()):
            nonfinite_parameters.append(name)
    if nonfinite_parameters:
        raise RuntimeError(
            "Non-finite parameters detected before training: "
            + ", ".join(nonfinite_parameters[:10])
        )

    normalization_modules = 0
    affine_normalization_scales = 0
    zero_normalization_scales: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, _NORMALIZATION_TYPES):
            continue
        normalization_modules += 1
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        affine_normalization_scales += 1
        if int(torch.count_nonzero(weight.detach()).cpu()) == 0:
            zero_normalization_scales.append(f"{name}.weight")
    if zero_normalization_scales:
        raise RuntimeError(
            "Zero-valued affine normalization scales detected before training. "
            "Do not blanket-reinitialize MONAI/PyTorch model parameters: "
            + ", ".join(zero_normalization_scales[:10])
        )

    head_audit = _audit_segmentation_head(model)
    return {
        "passed": True,
        "strategy": strategy,
        "blanket_parameter_override": False,
        "parameter_tensors_checked": parameter_tensors,
        "parameter_values_checked": parameter_values,
        "normalization_modules_checked": normalization_modules,
        "affine_normalization_scales_checked": affine_normalization_scales,
        "segmentation_head": head_audit,
    }
