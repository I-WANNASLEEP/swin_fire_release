"""Shared, testable utilities for ten-frame test-sequence visualizations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatial_models.swin_convlstm import SwinConvLSTM  # noqa: E402

AF_CHANNELS = (
    "I1_day",
    "I2_day",
    "I3_day",
    "I4_day",
    "I5_day",
    "M11_day",
    "I4_night",
    "I5_night",
)
VISUALIZATION_OUTPUT_SIZE = 224
AF_MEAN = np.asarray(
    [
        18.76488,
        27.441864,
        20.584806,
        305.99478,
        294.31738,
        14.625097,
        276.4207,
        275.16766,
    ],
    dtype=np.float32,
)
AF_STD = np.asarray(
    [
        15.911591,
        14.879259,
        10.832616,
        21.761852,
        24.703484,
        9.878246,
        40.64329,
        40.7657,
    ],
    dtype=np.float32,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_event_id(value: str) -> str:
    return value[3:] if value.startswith("af_") else value


def event_files(test_dir: Path, event_id: str) -> tuple[Path, Path]:
    event_id = canonical_event_id(event_id)
    image = test_dir / f"af_{event_id}_img_seqtoseql_10i_3.npy"
    label = test_dir / f"af_{event_id}_label_seqtoseql_10i_3.npy"
    if not image.is_file() or not label.is_file():
        available = sorted(
            path.name[3:].split("_img_", 1)[0]
            for path in test_dir.glob("af_*_img_seqtoseql_10i_3.npy")
        )
        raise FileNotFoundError(
            f"Missing test pair for event {event_id!r}. "
            f"Available examples: {available[:5]}"
        )
    return image, label


def assert_locked_test_event(event_id: str, split_file: Path) -> None:
    locked = {
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    event_id = canonical_event_id(event_id)
    if event_id not in locked:
        raise ValueError(
            f"Event {event_id!r} is not in the locked test split {split_file}."
        )


def load_test_window(
    test_dir: Path,
    event_id: str,
    window_index: int,
    *,
    label_channel: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_path, label_path = event_files(test_dir, event_id)
    images = np.load(image_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    if images.ndim != 5 or images.shape[1:] != (8, 10, 256, 256):
        raise ValueError(
            f"Expected images [N,8,10,256,256], received {images.shape}."
        )
    if labels.ndim != 5 or labels.shape[1] <= label_channel:
        raise ValueError(
            f"Expected labels [N,C,10,H,W] with channel {label_channel}, "
            f"received {labels.shape}."
        )
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Image and label files have different window counts.")
    if not 0 <= window_index < images.shape[0]:
        raise IndexError(
            f"window_index={window_index} is outside [0,{images.shape[0] - 1}]."
        )

    image = np.asarray(images[window_index], dtype=np.float32)
    raw_label = np.asarray(
        labels[window_index, label_channel], dtype=np.float32
    )
    if not np.isfinite(image).all() or not np.isfinite(raw_label).all():
        raise ValueError("Selected test window contains NaN or infinity.")
    if np.any((raw_label < 0) & (raw_label != -1)):
        raise ValueError("Selected label contains an unexpected negative value.")
    valid = raw_label != -1
    binary_label = (raw_label > 0) & valid
    metadata = {
        "event_id": canonical_event_id(event_id),
        "window_index": int(window_index),
        "window_count_in_event": int(images.shape[0]),
        "image_path": str(image_path.resolve()),
        "label_path": str(label_path.resolve()),
        "image_shape": list(image.shape),
        "label_shape": list(raw_label.shape),
        "label_channel": int(label_channel),
        "positive_label_rule": "target > 0",
        "ignore_index": -1,
        "valid_pixels": int(valid.sum()),
        "ignored_pixels": int((~valid).sum()),
        "fire_pixels": int(binary_label.sum()),
    }
    return image, raw_label, metadata


def normalized_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    normalized = (
        image - AF_MEAN[:, np.newaxis, np.newaxis, np.newaxis]
    ) / (AF_STD[:, np.newaxis, np.newaxis, np.newaxis] + 1e-8)
    return torch.from_numpy(normalized).unsqueeze(0).to(device)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def checkpoint_state(checkpoint_path: Path) -> tuple[dict[str, Any], dict]:
    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a mapping.")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break
    else:
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            state = payload
        else:
            raise KeyError("Checkpoint does not contain a model state dictionary.")
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return payload, state


def build_model(
    checkpoint_path: Path,
    *,
    attention: str,
    device: torch.device,
) -> tuple[SwinConvLSTM, dict, dict]:
    payload, state = checkpoint_state(checkpoint_path)
    patch_weight = state.get("swin_encoder.swinViT.patch_embed.proj.weight")
    lstm_weight = state.get("convlstm.conv.weight")
    if patch_weight is None or lstm_weight is None:
        raise KeyError("Checkpoint cannot establish SwinConvLSTM dimensions.")
    feature_size = int(patch_weight.shape[0])
    input_channels = int(patch_weight.shape[1])
    hidden_dim = int(lstm_weight.shape[0] // 4)
    if input_channels != 8:
        raise ValueError(
            f"This evaluator requires the paper's 8-channel model; "
            f"checkpoint has {input_channels} channels."
        )
    model = SwinConvLSTM(
        image_size=(256, 256),
        in_channels=input_channels,
        out_channels=2,
        feature_size=feature_size,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        hidden_dim=hidden_dim,
        dropout=0.1,
        attn_version=attention,
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    model.to(device)
    model.eval()
    model_metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_f1": _plain_number(payload.get("f1_score")),
        "attention": attention,
        "input_channels": input_channels,
        "feature_size": feature_size,
        "hidden_dim": hidden_dim,
        "device": str(device),
    }
    return model, payload, model_metadata


def _plain_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return value
    if np.isscalar(value):
        return value.item()
    return None


def frozen_threshold(payload: dict, override: float | None) -> float:
    if override is not None:
        threshold = float(override)
    else:
        candidates = (
            payload.get("frozen_threshold"),
            payload.get("best_threshold"),
            payload.get("val_metrics", {}).get("best_threshold")
            if isinstance(payload.get("val_metrics"), dict)
            else None,
        )
        threshold = next(
            (float(value) for value in candidates if value is not None), None
        )
        if threshold is None:
            raise ValueError(
                "Checkpoint has no frozen validation threshold. Supply "
                "--threshold only if it was frozen before test evaluation."
            )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [0,1].")
    return threshold


def infer_probabilities(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = normalized_tensor(image, device)
    with torch.inference_mode():
        logits = model(tensor)
        if logits.shape != (1, 2, 10, 256, 256):
            raise ValueError(
                "Expected model output [1,2,10,256,256], "
                f"received {tuple(logits.shape)}."
            )
        probabilities = torch.softmax(logits, dim=1)[:, 1]
    result = probabilities[0].detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Model produced non-finite probabilities.")
    return result


def frame_confusion(
    probabilities: np.ndarray,
    raw_label: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    prediction = probabilities >= threshold
    valid = raw_label != -1
    target = (raw_label > 0) & valid
    records = []
    for frame in range(10):
        p = prediction[frame] & valid[frame]
        y = target[frame]
        tp = int((p & y).sum())
        fp = int((p & ~y & valid[frame]).sum())
        fn = int((~p & y).sum())
        tn = int((~p & ~y & valid[frame]).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        records.append(
            {
                "frame_index": frame + 1,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
            }
        )
    return records


def center_crop_for_visualization(
    array: np.ndarray,
    output_size: int = VISUALIZATION_OUTPUT_SIZE,
) -> np.ndarray:
    """Return an aligned center crop over the final two spatial dimensions."""
    if array.ndim < 2:
        raise ValueError("Visualization array must have two spatial dimensions.")
    height, width = array.shape[-2:]
    if output_size <= 0 or output_size > min(height, width):
        raise ValueError(
            f"Invalid visualization output_size={output_size} for "
            f"spatial shape {(height, width)}."
        )
    top = (height - output_size) // 2
    left = (width - output_size) // 2
    return np.ascontiguousarray(
        array[..., top : top + output_size, left : left + output_size]
    )


def display_gray(frame: np.ndarray) -> np.ndarray:
    finite = frame[np.isfinite(frame)]
    low, high = np.percentile(finite, (2, 98))
    if high <= low:
        return np.zeros_like(frame, dtype=np.float32)
    return np.clip((frame - low) / (high - low), 0.0, 1.0).astype(
        np.float32
    )


def relative_attention_display(
    spatial_attention: np.ndarray,
    *,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    visible_threshold: float = 0.9,
    minimum_visible_alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Normalize attention for display and apply the locked opacity rule."""
    if not np.isfinite(spatial_attention).all():
        raise ValueError("Spatial attention contains NaN or infinity.")
    if not 0.0 <= visible_threshold < 1.0:
        raise ValueError("visible_threshold must be in [0,1).")
    if not 0.0 <= minimum_visible_alpha <= 1.0:
        raise ValueError("minimum_visible_alpha must be in [0,1].")
    display_low, display_high = np.percentile(
        spatial_attention, (percentile_low, percentile_high)
    )
    if display_high <= display_low:
        display_low = float(spatial_attention.min())
        display_high = float(spatial_attention.max())
    if display_high <= display_low:
        display_low, display_high = 0.0, 1.0
    relative = np.clip(
        (spatial_attention - display_low) / (display_high - display_low),
        0.0,
        1.0,
    )
    alpha = np.where(
        relative >= visible_threshold,
        minimum_visible_alpha
        + (1.0 - minimum_visible_alpha)
        * (relative - visible_threshold)
        / (1.0 - visible_threshold),
        0.0,
    )
    metadata: dict[str, float | str] = {
        "display_percentile_low": float(percentile_low),
        "display_percentile_high": float(percentile_high),
        "display_vmin": float(display_low),
        "display_vmax": float(display_high),
        "relative_visibility_threshold": float(visible_threshold),
        "minimum_visible_alpha": float(minimum_visible_alpha),
        "relative_alpha_rule": (
            f"alpha=0 below {visible_threshold}; alpha increases linearly "
            f"from {minimum_visible_alpha} at the threshold to 1.0"
        ),
        "relative_color_rule": (
            f"YlOrRd from yellow at {visible_threshold} to red at 1.0"
        ),
    }
    return relative.astype(np.float32), alpha.astype(np.float32), metadata


def confusion_overlay(
    base: np.ndarray,
    probability: np.ndarray,
    raw_label: np.ndarray,
    threshold: float,
    *,
    fn_color: str = "blue",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if fn_color not in {"green", "blue"}:
        raise ValueError("fn_color must be either 'green' or 'blue'.")
    valid = raw_label != -1
    target = (raw_label > 0) & valid
    prediction = (probability >= threshold) & valid
    masks = {
        "tp": prediction & target,
        "fp": prediction & ~target & valid,
        "fn": ~prediction & target,
    }
    gray = display_gray(base)
    rgb = np.repeat(gray[..., np.newaxis], 3, axis=-1)
    # Only pixels in TP/FP/FN are recolored. True-negative and ignored
    # background pixels remain the original grayscale image.
    fn_style = (
        (np.asarray([0.0, 0.38, 0.0]), 0.88)
        if fn_color == "green"
        else (np.asarray([0.0, 0.25, 1.0]), 0.82)
    )
    styles = {
        "tp": (np.asarray([0.0, 1.0, 0.0]), 0.62),
        "fp": (np.asarray([1.0, 0.0, 0.0]), 0.70),
        "fn": fn_style,
    }
    for key, mask in masks.items():
        color, alpha = styles[key]
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return np.clip(rgb, 0.0, 1.0), masks


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
