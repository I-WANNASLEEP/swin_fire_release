#!/usr/bin/env python3
"""Compare ten-frame fire detection and relative DCBAM attention in alignment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw

from _visualization_common import (
    AF_CHANNELS,
    PROJECT_ROOT,
    assert_locked_test_event,
    build_model,
    center_crop_for_visualization,
    confusion_overlay,
    display_gray,
    frame_confusion,
    frozen_threshold,
    load_test_window,
    relative_attention_display,
    select_device,
    write_json,
)
from visualize_attention_heatmap import capture_attention


def blend_attention(
    base: np.ndarray,
    relative_attention: np.ndarray,
    attention_alpha: np.ndarray,
) -> np.ndarray:
    gray = display_gray(base)
    gray_rgb = np.repeat(gray[..., np.newaxis], 3, axis=-1)
    colors = plt.get_cmap("YlOrRd")(
        Normalize(vmin=0.9, vmax=1.0, clip=True)(relative_attention)
    )[..., :3]
    alpha_rgb = attention_alpha[..., np.newaxis]
    return np.clip(
        (1.0 - alpha_rgb) * gray_rgb + alpha_rgb * colors,
        0.0,
        1.0,
    )


def blend_overlap(
    base: np.ndarray,
    target: np.ndarray,
    attention_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gray = display_gray(base)
    rgb = np.repeat(gray[..., np.newaxis], 3, axis=-1)
    masks = {
        "fire_only": target & ~attention_mask,
        "attention_only": attention_mask & ~target,
        "overlap": target & attention_mask,
    }
    styles = {
        "fire_only": (np.asarray([0.0, 1.0, 0.0]), 0.82),
        "attention_only": (np.asarray([1.0, 0.48, 0.0]), 0.68),
        "overlap": (np.asarray([1.0, 0.0, 1.0]), 0.92),
    }
    for key, mask in masks.items():
        color, alpha = styles[key]
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return np.clip(rgb, 0.0, 1.0), masks


def overlap_record(
    target: np.ndarray,
    prediction: np.ndarray,
    attention_mask: np.ndarray,
    valid: np.ndarray,
    frame_index: int,
) -> dict:
    attention = attention_mask & valid
    target = target & valid
    prediction = prediction & valid

    def compare(reference: np.ndarray) -> dict[str, float | int]:
        intersection = int((reference & attention).sum())
        reference_count = int(reference.sum())
        attention_count = int(attention.sum())
        union = reference_count + attention_count - intersection
        return {
            "intersection_pixels": intersection,
            "reference_pixels": reference_count,
            "attention_pixels": attention_count,
            "reference_coverage": (
                intersection / reference_count if reference_count else 0.0
            ),
            "attention_precision": (
                intersection / attention_count if attention_count else 0.0
            ),
            "iou": intersection / union if union else 0.0,
            "dice": (
                2 * intersection / (reference_count + attention_count)
                if reference_count + attention_count
                else 0.0
            ),
        }

    return {
        "frame_index": frame_index,
        "ground_truth_vs_attention": compare(target),
        "prediction_vs_attention": compare(prediction),
    }


def pooled_overlap(records: list[dict], key: str) -> dict[str, float | int]:
    intersection = sum(
        row[key]["intersection_pixels"] for row in records
    )
    reference = sum(row[key]["reference_pixels"] for row in records)
    attention = sum(row[key]["attention_pixels"] for row in records)
    union = reference + attention - intersection
    return {
        "intersection_pixels": intersection,
        "reference_pixels": reference,
        "attention_pixels": attention,
        "reference_coverage": intersection / reference if reference else 0.0,
        "attention_precision": intersection / attention if attention else 0.0,
        "iou": intersection / union if union else 0.0,
        "dice": (
            2 * intersection / (reference + attention)
            if reference + attention
            else 0.0
        ),
    }


def render_pages(
    image: np.ndarray,
    raw_label: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    relative_attention: np.ndarray,
    attention_alpha: np.ndarray,
    channel_index: int,
    detection_records: list[dict],
    overlap_records: list[dict],
    output_dir: Path,
) -> None:
    for start in (0, 5):
        figure, axes = plt.subplots(
            5, 5, figsize=(19, 20), constrained_layout=True
        )
        for row, frame_index in enumerate(range(start, start + 5)):
            valid = raw_label[frame_index] != -1
            target = (raw_label[frame_index] > 0) & valid
            prediction = (
                probabilities[frame_index] >= threshold
            ) & valid
            fire_overlay, fire_masks = confusion_overlay(
                image[channel_index, frame_index],
                probabilities[frame_index],
                raw_label[frame_index],
                threshold,
                fn_color="green",
            )
            attention_overlay = blend_attention(
                image[channel_index, frame_index],
                relative_attention[frame_index],
                attention_alpha[frame_index],
            )
            attention_mask = (attention_alpha[frame_index] > 0.0) & valid
            overlap_overlay, _overlap_masks = blend_overlap(
                image[channel_index, frame_index],
                target,
                attention_mask,
            )
            panels = (
                (
                    display_gray(image[channel_index, frame_index]),
                    "gray",
                    f"Input ({AF_CHANNELS[channel_index]}, T={frame_index})",
                ),
                (target, "gray", "Ground Truth Fire"),
                (
                    fire_overlay,
                    None,
                    f"Detection (th={threshold:.2f})",
                ),
                (
                    attention_overlay,
                    None,
                    "Relative Attention (>=0.9)",
                ),
                (
                    overlap_overlay,
                    None,
                    "GT–Attention Overlap",
                ),
            )
            for column, (panel, cmap, title) in enumerate(panels):
                axis = axes[row, column]
                axis.imshow(
                    panel,
                    cmap=cmap,
                    vmin=0 if cmap else None,
                    vmax=1 if cmap else None,
                    interpolation="nearest",
                )
                if column == 2 and fire_masks["fn"].any():
                    axis.contour(
                        fire_masks["fn"].astype(np.uint8),
                        levels=[0.5],
                        colors=["#b6ff00"],
                        linewidths=0.7,
                    )
                axis.set_title(title, fontsize=10)
                axis.axis("off")
            detection = detection_records[frame_index]
            alignment = overlap_records[frame_index][
                "ground_truth_vs_attention"
            ]
            axes[row, 4].set_xlabel(
                f"Frame {frame_index + 1}/10 | detection F1="
                f"{detection['f1']:.3f}\n"
                f"fire coverage={alignment['reference_coverage']:.3f} | "
                f"attention precision={alignment['attention_precision']:.3f} | "
                f"IoU={alignment['iou']:.3f}",
                fontsize=8,
            )
        figure.suptitle(
            "Aligned fire detection and temporal relative attention "
            f"(frames {start + 1}-{start + 5})",
            fontsize=15,
        )
        figure.savefig(
            output_dir
            / f"fire_attention_comparison_frames_{start + 1:02d}_{start + 5:02d}.png",
            dpi=180,
        )
        plt.close(figure)


def render_gif(
    image: np.ndarray,
    raw_label: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    relative_attention: np.ndarray,
    attention_alpha: np.ndarray,
    channel_index: int,
    overlap_records: list[dict],
    output_dir: Path,
) -> None:
    frames = []
    for frame_index in range(10):
        valid = raw_label[frame_index] != -1
        target = (raw_label[frame_index] > 0) & valid
        attention_mask = (attention_alpha[frame_index] > 0.0) & valid
        fire_overlay, _ = confusion_overlay(
            image[channel_index, frame_index],
            probabilities[frame_index],
            raw_label[frame_index],
            threshold,
            fn_color="green",
        )
        attention_overlay = blend_attention(
            image[channel_index, frame_index],
            relative_attention[frame_index],
            attention_alpha[frame_index],
        )
        overlap_overlay, _ = blend_overlap(
            image[channel_index, frame_index], target, attention_mask
        )
        panels = []
        for array, label in (
            (fire_overlay, "Detection TP/FP/FN"),
            (attention_overlay, "Relative attention >= 0.9"),
            (overlap_overlay, "GT-attention overlap"),
        ):
            panel = Image.fromarray(
                (array * 255).astype(np.uint8)
            ).resize((384, 384))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 384, 30), fill=(0, 0, 0))
            draw.text((8, 8), label, fill=(255, 255, 255))
            panels.append(panel)
        canvas = Image.new("RGB", (1152, 426), color=(255, 255, 255))
        for index, panel in enumerate(panels):
            canvas.paste(panel, (index * 384, 42))
        record = overlap_records[frame_index]["ground_truth_vs_attention"]
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 12),
            f"Frame {frame_index + 1}/10 | fire coverage="
            f"{record['reference_coverage']:.3f} | attention precision="
            f"{record['attention_precision']:.3f} | IoU={record['iou']:.3f}",
            fill=(0, 0, 0),
        )
        frames.append(canvas)
    frames[0].save(
        output_dir / "fire_attention_comparison.gif",
        save_all=True,
        append_images=frames[1:],
        duration=800,
        loop=0,
    )


def write_overlap_csv(
    path: Path,
    detection_records: list[dict],
    overlap_records: list[dict],
) -> None:
    fields = [
        "frame_index",
        "detection_f1",
        "detection_iou",
        "gt_fire_pixels",
        "relative_attention_pixels",
        "intersection_pixels",
        "fire_coverage",
        "attention_fire_precision",
        "attention_fire_iou",
        "attention_fire_dice",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detection, overlap in zip(
            detection_records, overlap_records, strict=True
        ):
            ground_truth = overlap["ground_truth_vs_attention"]
            writer.writerow(
                {
                    "frame_index": detection["frame_index"],
                    "detection_f1": detection["f1"],
                    "detection_iou": detection["iou"],
                    "gt_fire_pixels": ground_truth["reference_pixels"],
                    "relative_attention_pixels": ground_truth[
                        "attention_pixels"
                    ],
                    "intersection_pixels": ground_truth[
                        "intersection_pixels"
                    ],
                    "fire_coverage": ground_truth["reference_coverage"],
                    "attention_fire_precision": ground_truth[
                        "attention_precision"
                    ],
                    "attention_fire_iou": ground_truth["iou"],
                    "attention_fire_dice": ground_truth["dice"],
                }
            )


def render_overlap_metrics(
    detection_records: list[dict],
    overlap_records: list[dict],
    pooled_ground_truth: dict,
    output_dir: Path,
) -> None:
    frames = np.arange(1, 11)
    detection_f1 = [row["f1"] for row in detection_records]
    coverage = [
        row["ground_truth_vs_attention"]["reference_coverage"]
        for row in overlap_records
    ]
    precision = [
        row["ground_truth_vs_attention"]["attention_precision"]
        for row in overlap_records
    ]
    iou = [
        row["ground_truth_vs_attention"]["iou"]
        for row in overlap_records
    ]
    figure, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    axis.plot(frames, detection_f1, marker="o", label="Detection F1")
    axis.plot(frames, coverage, marker="o", label="Fire coverage by attention")
    axis.plot(
        frames,
        precision,
        marker="o",
        label="Attention pixels on fire",
    )
    axis.plot(frames, iou, marker="o", label="Fire–attention IoU")
    axis.set(
        xlabel="Frame",
        ylabel="Metric value",
        xticks=frames,
        ylim=(-0.02, 1.02),
        title="Temporal fire detection and relative-attention alignment",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="center right")
    axis.text(
        0.015,
        0.03,
        "Pooled GT comparison: "
        f"coverage={pooled_ground_truth['reference_coverage']:.4f}, "
        f"attention precision={pooled_ground_truth['attention_precision']:.4f}, "
        f"IoU={pooled_ground_truth['iou']:.4f}",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    figure.savefig(
        output_dir / "fire_attention_overlap_metrics.png", dpi=180
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-dir", type=Path, default=PROJECT_ROOT / "dataset_test"
    )
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--attention", choices=("cbam", "dcbam"), default="dcbam")
    parser.add_argument(
        "--display-channel", choices=AF_CHANNELS, default="I4_day"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    split_file = PROJECT_ROOT / "splits" / "test_event_ids.txt"
    assert_locked_test_event(args.event_id, split_file)
    image, raw_label, data_metadata = load_test_window(
        args.test_dir.expanduser().resolve(),
        args.event_id,
        args.window_index,
    )
    device = select_device(args.device)
    model, payload, model_metadata = build_model(
        args.checkpoint.expanduser().resolve(),
        attention=args.attention,
        device=device,
    )
    threshold = frozen_threshold(payload, None)
    (
        spatial_pre_sigmoid,
        spatial_attention,
        channel_attention,
        probabilities,
    ) = capture_attention(model, image, device)
    image = center_crop_for_visualization(image)
    raw_label = center_crop_for_visualization(raw_label)
    probabilities = center_crop_for_visualization(probabilities)
    spatial_pre_sigmoid = center_crop_for_visualization(spatial_pre_sigmoid)
    spatial_attention = center_crop_for_visualization(spatial_attention)
    relative_attention, attention_alpha, relative_metadata = (
        relative_attention_display(spatial_attention)
    )
    detection_records = frame_confusion(
        probabilities, raw_label, threshold
    )
    prediction = probabilities >= threshold
    valid = raw_label != -1
    target = (raw_label > 0) & valid
    attention_mask = (attention_alpha > 0.0) & valid
    overlap_records = [
        overlap_record(
            target[frame],
            prediction[frame],
            attention_mask[frame],
            valid[frame],
            frame + 1,
        )
        for frame in range(10)
    ]
    pooled_ground_truth = pooled_overlap(
        overlap_records, "ground_truth_vs_attention"
    )
    pooled_prediction = pooled_overlap(
        overlap_records, "prediction_vs_attention"
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "results"
        / "fire_attention_alignment"
        / f"{data_metadata['event_id']}_window_{args.window_index:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    channel_index = AF_CHANNELS.index(args.display_channel)
    render_pages(
        image,
        raw_label,
        probabilities,
        threshold,
        relative_attention,
        attention_alpha,
        channel_index,
        detection_records,
        overlap_records,
        output_dir,
    )
    render_gif(
        image,
        raw_label,
        probabilities,
        threshold,
        relative_attention,
        attention_alpha,
        channel_index,
        overlap_records,
        output_dir,
    )
    write_overlap_csv(
        output_dir / "fire_attention_overlap_metrics.csv",
        detection_records,
        overlap_records,
    )
    render_overlap_metrics(
        detection_records,
        overlap_records,
        pooled_ground_truth,
        output_dir,
    )
    np.savez_compressed(
        output_dir / "fire_attention_alignment_arrays.npz",
        fire_probability=probabilities,
        ground_truth_fire=target,
        spatial_attention_pre_sigmoid=spatial_pre_sigmoid,
        spatial_attention=spatial_attention,
        relative_attention=relative_attention,
        attention_alpha=attention_alpha,
        relative_attention_mask=attention_mask,
        latent_channel_attention=channel_attention,
    )
    write_json(
        output_dir / "fire_attention_alignment_metrics.json",
        {
            "data": data_metadata,
            "model": model_metadata,
            "frozen_prediction_threshold": threshold,
            "visualization_crop": {
                "model_input": "256x256",
                "display_output": "224x224",
                "source_slice": "[...,16:240,16:240]",
                "scope": "visualization and visualization-alignment metrics only",
            },
            "attention_display": relative_metadata,
            "frame_detection_metrics": detection_records,
            "frame_overlap_metrics": overlap_records,
            "pooled_ground_truth_vs_attention": pooled_ground_truth,
            "pooled_prediction_vs_attention": pooled_prediction,
            "overlap_colors": {
                "ground_truth_fire_only": "green",
                "relative_attention_only": "orange",
                "ground_truth_and_relative_attention": "magenta",
            },
            "interpretation_limit": (
                "The relative-attention threshold is a user-selected display "
                "rule, not a validation-selected segmentation threshold. "
                "Overlap values describe visualization alignment and are not "
                "model accuracy or causal attribution metrics."
            ),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
