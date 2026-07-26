#!/usr/bin/env python3
"""Render one locked TS-SatFire test window from frame 1 through frame 10."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

from _visualization_common import (
    AF_CHANNELS,
    PROJECT_ROOT,
    assert_locked_test_event,
    build_model,
    confusion_overlay,
    display_gray,
    frame_confusion,
    frozen_threshold,
    infer_probabilities,
    load_test_window,
    select_device,
    write_json,
)


def render(
    image: np.ndarray,
    raw_label: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    channel_index: int,
    output_dir: Path,
    records: list[dict],
    fn_color: str,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    gif_frames = []
    for frame_index, axis in enumerate(axes.flat):
        overlay, masks = confusion_overlay(
            image[channel_index, frame_index],
            probabilities[frame_index],
            raw_label[frame_index],
            threshold,
            fn_color=fn_color,
        )
        axis.imshow(overlay, interpolation="nearest")
        if fn_color == "green" and masks["fn"].any():
            axis.contour(
                masks["fn"].astype(np.uint8),
                levels=[0.5],
                colors=["#b6ff00"],
                linewidths=0.7,
            )
        record = records[frame_index]
        axis.set_title(
            f"Frame {frame_index + 1}  "
            f"F1={record['f1']:.3f}  IoU={record['iou']:.3f}"
        )
        axis.axis("off")

        frame_rgb = (overlay * 255).astype(np.uint8)
        frame_image = Image.fromarray(frame_rgb).resize((512, 512))
        draw = ImageDraw.Draw(frame_image)
        draw.rectangle((0, 0, 512, 34), fill=(0, 0, 0))
        draw.text(
            (10, 9),
            f"Frame {frame_index + 1}/10  F1={record['f1']:.3f}  "
            f"IoU={record['iou']:.3f}",
            fill=(255, 255, 255),
        )
        gif_frames.append(frame_image)

    figure.suptitle(
        f"10-frame test sequence | base={AF_CHANNELS[channel_index]} | "
        f"frozen threshold={threshold:.2f}",
        fontsize=15,
    )
    figure.legend(
        handles=[
            Patch(facecolor="#00ff00", label="TP (green)"),
            Patch(facecolor="#ff0000", label="FP (red)"),
            Patch(
                facecolor="#006100" if fn_color == "green" else "#0040ff",
                edgecolor="#b6ff00" if fn_color == "green" else "#0040ff",
                label=(
                    "FN (dark green + bright-green outline)"
                    if fn_color == "green"
                    else "FN (blue)"
                ),
            ),
        ],
        loc="lower center",
        ncol=3,
    )
    figure.savefig(output_dir / "sequence_tp_fp_fn.png", dpi=180)
    plt.close(figure)
    gif_frames[0].save(
        output_dir / "sequence_tp_fp_fn.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=700,
        loop=0,
    )
    render_reference_pages(
        image,
        raw_label,
        probabilities,
        threshold,
        channel_index,
        output_dir,
        records,
        fn_color,
    )


def render_reference_pages(
    image: np.ndarray,
    raw_label: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    channel_index: int,
    output_dir: Path,
    records: list[dict],
    fn_color: str,
) -> None:
    """Render the user's four-column diagnostic layout for all ten frames."""
    fn_label = "Green" if fn_color == "green" else "Blue"
    for page_index, start in enumerate((0, 5), start=1):
        figure, axes = plt.subplots(
            5,
            4,
            figsize=(16, 20),
            constrained_layout=True,
        )
        for row, frame_index in enumerate(range(start, start + 5)):
            valid = raw_label[frame_index] != -1
            target = (raw_label[frame_index] > 0) & valid
            prediction = (
                probabilities[frame_index] >= threshold
            ) & valid
            overlay, masks = confusion_overlay(
                image[channel_index, frame_index],
                probabilities[frame_index],
                raw_label[frame_index],
                threshold,
                fn_color=fn_color,
            )
            base = display_gray(image[channel_index, frame_index])
            panels = (
                (base, "gray", f"Input ({AF_CHANNELS[channel_index]}, T={frame_index})"),
                (target, "gray", "Ground Truth"),
                (
                    prediction,
                    "gray",
                    f"Prediction (th={threshold:.2f})",
                ),
                (
                    overlay,
                    None,
                    f"TP(Green)/FP(Red)/FN({fn_label})",
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
                if (
                    column == 3
                    and fn_color == "green"
                    and masks["fn"].any()
                ):
                    axis.contour(
                        masks["fn"].astype(np.uint8),
                        levels=[0.5],
                        colors=["#b6ff00"],
                        linewidths=0.7,
                    )
                axis.set_title(title, fontsize=10)
                axis.axis("off")
            record = records[frame_index]
            axes[row, 3].set_xlabel(
                f"Frame {frame_index + 1}/10 | "
                f"F1={record['f1']:.3f} | IoU={record['iou']:.3f}",
                fontsize=9,
            )
        figure.suptitle(
            "10-frame fire evolution diagnostic "
            f"(frames {start + 1}-{start + 5})",
            fontsize=15,
        )
        figure.savefig(
            output_dir
            / f"sequence_diagnostic_frames_{start + 1:02d}_{start + 5:02d}.png",
            dpi=180,
        )
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_test",
    )
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--attention", default="dcbam")
    parser.add_argument(
        "--display-channel",
        choices=AF_CHANNELS,
        default="I4_day",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--fn-color",
        choices=("green", "blue"),
        default="green",
        help=(
            "Use green to follow the requested manuscript convention, or blue "
            "to reproduce the supplied reference figure exactly."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    test_dir = args.test_dir.expanduser().resolve()
    split_file = PROJECT_ROOT / "splits" / "test_event_ids.txt"
    assert_locked_test_event(args.event_id, split_file)
    image, raw_label, data_metadata = load_test_window(
        test_dir, args.event_id, args.window_index
    )
    device = select_device(args.device)
    model, checkpoint_payload, model_metadata = build_model(
        checkpoint, attention=args.attention, device=device
    )
    threshold = frozen_threshold(checkpoint_payload, args.threshold)
    probabilities = infer_probabilities(model, image, device)
    records = frame_confusion(probabilities, raw_label, threshold)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "results"
        / "visualizations"
        / f"{data_metadata['event_id']}_window_{args.window_index:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    channel_index = AF_CHANNELS.index(args.display_channel)
    render(
        image,
        raw_label,
        probabilities,
        threshold,
        channel_index,
        output_dir,
        records,
        args.fn_color,
    )
    np.save(output_dir / "fire_probabilities.npy", probabilities)
    write_json(
        output_dir / "sequence_metrics.json",
        {
            "data": data_metadata,
            "model": model_metadata,
            "threshold": threshold,
            "threshold_source": (
                "explicit_frozen_override"
                if args.threshold is not None
                else "checkpoint"
            ),
            "frame_metrics": records,
            "temporal_axis": (
                "sequence indices 1-10; physical timestamps are unavailable "
                "because the provided arrays have no window-time manifest"
            ),
            "overlay": {
                "tp": "bright green",
                "fp": "red",
                "fn": (
                    "dark green with bright-green outline"
                    if args.fn_color == "green"
                    else "blue"
                ),
                "coverage": (
                    "only TP, FP, and FN pixels are recolored; all other "
                    "pixels retain the input grayscale"
                ),
            },
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
