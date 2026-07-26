#!/usr/bin/env python3
"""Render every requested test window into one flat, PNG-only directory."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from _visualization_common import (
    AF_CHANNELS,
    PROJECT_ROOT,
    assert_locked_test_event,
    build_model,
    center_crop_for_visualization,
    event_files,
    frame_confusion,
    frozen_threshold,
    load_test_window,
    relative_attention_display,
    select_device,
)
from visualize_attention_heatmap import (
    capture_attention,
    render as render_attention,
)
from visualize_fire_attention_alignment import (
    overlap_record,
    pooled_overlap,
    render_overlap_metrics,
    render_pages as render_alignment_pages,
)
from visualize_test_sequence import render as render_detection


PNG_PRODUCTS = {
    "fire_monitoring/sequence_tp_fp_fn.png": "fire_monitoring_overview.png",
    (
        "fire_monitoring/sequence_diagnostic_frames_01_05.png"
    ): "fire_monitoring_frames_01_05.png",
    (
        "fire_monitoring/sequence_diagnostic_frames_06_10.png"
    ): "fire_monitoring_frames_06_10.png",
    "attention/attention_heatmap.png": "attention_frames_01_10.png",
    (
        "fire_attention_alignment/fire_attention_comparison_frames_01_05.png"
    ): "comparison_frames_01_05.png",
    (
        "fire_attention_alignment/fire_attention_comparison_frames_06_10.png"
    ): "comparison_frames_06_10.png",
    (
        "fire_attention_alignment/fire_attention_overlap_metrics.png"
    ): "overlap_metrics.png",
}


def locked_event_ids(split_file: Path) -> list[str]:
    event_ids = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not event_ids:
        raise ValueError(f"No event IDs found in {split_file}.")
    return event_ids


def window_count(test_dir: Path, event_id: str) -> int:
    image_path, label_path = event_files(test_dir, event_id)
    images = np.load(image_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    if images.ndim != 5 or labels.ndim != 5:
        raise ValueError(f"Unexpected array rank for event {event_id}.")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Image/label window-count mismatch for {event_id}.")
    return int(images.shape[0])


def flat_png_paths(
    output_root: Path,
    event_id: str,
    window_index: int,
) -> dict[str, Path]:
    prefix = f"{event_id}__window_{window_index:03d}__"
    return {
        source: output_root / f"{prefix}{suffix}"
        for source, suffix in PNG_PRODUCTS.items()
    }


def validate_flat_png_directory(output_root: Path) -> None:
    if not output_root.exists():
        return
    invalid = [
        path
        for path in output_root.iterdir()
        if not path.is_file() or path.suffix.lower() != ".png"
    ]
    if invalid:
        preview = ", ".join(path.name for path in invalid[:5])
        raise ValueError(
            f"{output_root} must contain PNG files only; found: {preview}"
        )


def selected_window_indices(
    count: int,
    requested: list[int] | None,
    max_windows: int | None,
) -> list[int]:
    if requested is not None:
        invalid = [index for index in requested if not 0 <= index < count]
        if invalid:
            raise IndexError(
                f"Window indices {invalid} are outside [0,{count - 1}]."
            )
        return sorted(set(requested))
    limit = min(count, max_windows) if max_windows is not None else count
    return list(range(limit))


def render_window_pngs(
    *,
    image: np.ndarray,
    raw_label: np.ndarray,
    probabilities: np.ndarray,
    spatial_attention: np.ndarray,
    threshold: float,
    channel_index: int,
    temporary_root: Path,
) -> None:
    image = center_crop_for_visualization(image)
    raw_label = center_crop_for_visualization(raw_label)
    probabilities = center_crop_for_visualization(probabilities)
    spatial_attention = center_crop_for_visualization(spatial_attention)

    detection_dir = temporary_root / "fire_monitoring"
    attention_dir = temporary_root / "attention"
    alignment_dir = temporary_root / "fire_attention_alignment"
    detection_dir.mkdir()
    attention_dir.mkdir()
    alignment_dir.mkdir()

    detection_records = frame_confusion(probabilities, raw_label, threshold)
    render_detection(
        image,
        raw_label,
        probabilities,
        threshold,
        channel_index,
        detection_dir,
        detection_records,
        "green",
        write_gif=False,
    )
    render_attention(
        image,
        spatial_attention,
        channel_index,
        attention_dir,
        write_gif=False,
    )

    relative_attention, attention_alpha, _ = relative_attention_display(
        spatial_attention
    )
    valid = raw_label != -1
    target = (raw_label > 0) & valid
    prediction = (probabilities >= threshold) & valid
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
    render_alignment_pages(
        image,
        raw_label,
        probabilities,
        threshold,
        relative_attention,
        attention_alpha,
        channel_index,
        detection_records,
        overlap_records,
        alignment_dir,
    )
    render_overlap_metrics(
        detection_records,
        overlap_records,
        pooled_ground_truth,
        alignment_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-dir", type=Path, default=PROJECT_ROOT / "dataset_test"
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "splits" / "test_event_ids.txt",
    )
    parser.add_argument(
        "--event-id",
        action="append",
        default=None,
        help="Restrict to one event; repeat for multiple events.",
    )
    parser.add_argument(
        "--window-index",
        action="append",
        type=int,
        default=None,
        help=(
            "Restrict to a zero-based window; repeat for multiple windows. "
            "This option requires exactly one --event-id."
        ),
    )
    parser.add_argument(
        "--max-windows-per-event",
        type=int,
        default=None,
        help="Smoke-test limit. Omit to render every window.",
    )
    parser.add_argument(
        "--attention", choices=("cbam", "dcbam"), default="dcbam"
    )
    parser.add_argument(
        "--display-channel", choices=AF_CHANNELS, default="I4_day"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip windows whose seven flat PNG products already exist.",
    )
    args = parser.parse_args()

    if (
        args.max_windows_per_event is not None
        and args.max_windows_per_event <= 0
    ):
        parser.error("--max-windows-per-event must be positive.")
    if args.window_index is not None and (
        args.event_id is None or len(args.event_id) != 1
    ):
        parser.error("--window-index requires exactly one --event-id.")
    if (
        args.window_index is not None
        and args.max_windows_per_event is not None
    ):
        parser.error(
            "--window-index and --max-windows-per-event cannot be combined."
        )

    test_dir = args.test_dir.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    event_ids = args.event_id or locked_event_ids(split_file)
    for event_id in event_ids:
        assert_locked_test_event(event_id, split_file)

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"{output_root} already exists. Use a new path or --resume."
        )
    validate_flat_png_directory(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    counts = {
        event_id: window_count(test_dir, event_id)
        for event_id in event_ids
    }
    windows = {
        event_id: selected_window_indices(
            counts[event_id],
            args.window_index if len(event_ids) == 1 else None,
            args.max_windows_per_event,
        )
        for event_id in event_ids
    }
    planned = sum(len(indices) for indices in windows.values())
    device = select_device(args.device)
    model, payload, _model_metadata = build_model(
        args.checkpoint.expanduser().resolve(),
        attention=args.attention,
        device=device,
    )
    threshold = frozen_threshold(payload, None)
    channel_index = AF_CHANNELS.index(args.display_channel)

    completed = 0
    for event_id in event_ids:
        for window_index in windows[event_id]:
            destinations = flat_png_paths(
                output_root, event_id, window_index
            )
            all_products_exist = all(
                path.is_file() for path in destinations.values()
            )
            if args.resume and all_products_exist:
                completed += 1
                print(
                    f"[{completed}/{planned}] skip {event_id} "
                    f"window {window_index}"
                )
                continue
            print(
                f"[{completed + 1}/{planned}] render {event_id} "
                f"window {window_index}"
            )
            image, raw_label, _data_metadata = load_test_window(
                test_dir, event_id, window_index
            )
            (
                _spatial_pre_sigmoid,
                spatial_attention,
                _channel_attention,
                probabilities,
            ) = capture_attention(model, image, device)
            with tempfile.TemporaryDirectory(
                prefix="swin_fire_png_"
            ) as temporary_directory:
                temporary_root = Path(temporary_directory)
                render_window_pngs(
                    image=image,
                    raw_label=raw_label,
                    probabilities=probabilities,
                    spatial_attention=spatial_attention,
                    threshold=threshold,
                    channel_index=channel_index,
                    temporary_root=temporary_root,
                )
                for source_name, destination in destinations.items():
                    source = temporary_root / source_name
                    if not source.is_file():
                        raise FileNotFoundError(
                            f"Expected renderer output is missing: {source}"
                        )
                    if not destination.exists():
                        shutil.copy2(source, destination)
            completed += 1

    validate_flat_png_directory(output_root)
    print(
        f"Completed {completed}/{planned} windows; "
        f"{len(list(output_root.glob('*.png')))} PNG files in {output_root}"
    )


if __name__ == "__main__":
    main()
