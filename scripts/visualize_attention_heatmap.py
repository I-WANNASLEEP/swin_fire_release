#!/usr/bin/env python3
"""Export the model's actual CBAM/DCBAM attention as ten-frame heatmaps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw

from _visualization_common import (
    AF_CHANNELS,
    PROJECT_ROOT,
    assert_locked_test_event,
    build_model,
    display_gray,
    load_test_window,
    normalized_tensor,
    relative_attention_display,
    select_device,
    write_json,
)


def capture_attention(model, image, device):
    attention = model.attention
    if not hasattr(attention, "conv_spatial") or not hasattr(attention, "fc"):
        raise ValueError(
            "Exact spatial heatmaps require a CBAM or DCBAM checkpoint; "
            "SE and identity attention have no spatial attention tensor."
        )
    spatial_logits = []
    channel_logits = []

    spatial_hook = attention.conv_spatial.register_forward_hook(
        lambda _module, _inputs, output: spatial_logits.append(
            output.detach().cpu()
        )
    )
    channel_hook = attention.fc.register_forward_hook(
        lambda _module, _inputs, output: channel_logits.append(
            output.detach().cpu()
        )
    )
    try:
        with torch.inference_mode():
            logits = model(normalized_tensor(image, device))
            probabilities = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
    finally:
        spatial_hook.remove()
        channel_hook.remove()

    if len(spatial_logits) != 10 or len(channel_logits) != 20:
        raise RuntimeError(
            "Unexpected attention-hook count: "
            f"spatial={len(spatial_logits)}, channel={len(channel_logits)}."
        )
    spatial_pre_sigmoid = np.stack(
        [value[0, 0].numpy() for value in spatial_logits]
    )
    spatial = np.stack(
        [torch.sigmoid(value)[0, 0].numpy() for value in spatial_logits]
    )
    channel = np.stack(
        [
            torch.sigmoid(channel_logits[2 * index] + channel_logits[2 * index + 1])
            .flatten()
            .numpy()
            for index in range(10)
        ]
    )
    return spatial_pre_sigmoid, spatial, channel, probabilities


def render(image, spatial, channel_index, output_dir):
    relative_attention, attention_alpha, relative_metadata = (
        relative_attention_display(spatial)
    )
    visible_threshold = float(
        relative_metadata["relative_visibility_threshold"]
    )
    color_norm = Normalize(vmin=visible_threshold, vmax=1.0, clip=True)
    figure, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    gif_frames = []
    for frame_index, axis in enumerate(axes.flat):
        gray = display_gray(image[channel_index, frame_index])
        axis.imshow(gray, cmap="gray", vmin=0, vmax=1)
        relative_frame = relative_attention[frame_index]
        alpha_map = attention_alpha[frame_index]
        heatmap = axis.imshow(
            relative_frame,
            cmap="YlOrRd",
            norm=color_norm,
            alpha=alpha_map,
            interpolation="nearest",
        )
        axis.set_title(
            f"Frame {frame_index + 1}  "
            f"mean={spatial[frame_index].mean():.3f}  "
            f"std={spatial[frame_index].std():.3f}"
        )
        axis.axis("off")

        colored = plt.get_cmap("YlOrRd")(
            color_norm(relative_frame)
        )[..., :3]
        alpha_rgb = alpha_map[..., np.newaxis]
        overlay = (
            (1.0 - alpha_rgb) * np.repeat(gray[..., None], 3, axis=-1)
            + alpha_rgb * colored
        )
        frame_image = Image.fromarray(
            (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
        ).resize((512, 512))
        draw = ImageDraw.Draw(frame_image)
        draw.rectangle((0, 0, 512, 34), fill=(0, 0, 0))
        draw.text(
            (10, 9),
            f"Frame {frame_index + 1}/10  attention mean="
            f"{spatial[frame_index].mean():.3f}",
            fill=(255, 255, 255),
        )
        gif_frames.append(frame_image)

    figure.suptitle(
        "Relative-contrast DCBAM/CBAM attention overlay "
        f"| base={AF_CHANNELS[channel_index]} | visible relative score >= 0.9",
        fontsize=15,
    )
    colorbar = figure.colorbar(
        heatmap, ax=axes.ravel().tolist(), shrink=0.72
    )
    colorbar.set_label(
        "Relative attention (p2-p98 normalized; <0.9 transparent)"
    )
    figure.savefig(output_dir / "attention_heatmap.png", dpi=180)
    plt.close(figure)
    gif_frames[0].save(
        output_dir / "attention_heatmap.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=700,
        loop=0,
    )
    return {
        "absolute_min": float(spatial.min()),
        "absolute_max": float(spatial.max()),
        "absolute_mean": float(spatial.mean()),
        "absolute_std": float(spatial.std()),
        "fraction_ge_0_80": float((spatial >= 0.80).mean()),
        "fraction_ge_0_90": float((spatial >= 0.90).mean()),
        "fraction_ge_0_95": float((spatial >= 0.95).mean()),
        "fraction_ge_0_98": float((spatial >= 0.98).mean()),
        **relative_metadata,
    }


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
    parser.add_argument(
        "--attention", choices=("cbam", "dcbam"), default="dcbam"
    )
    parser.add_argument(
        "--display-channel",
        choices=AF_CHANNELS,
        default="I4_day",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    split_file = PROJECT_ROOT / "splits" / "test_event_ids.txt"
    assert_locked_test_event(args.event_id, split_file)
    image, _raw_label, data_metadata = load_test_window(
        args.test_dir.expanduser().resolve(),
        args.event_id,
        args.window_index,
    )
    device = select_device(args.device)
    model, _payload, model_metadata = build_model(
        args.checkpoint.expanduser().resolve(),
        attention=args.attention,
        device=device,
    )
    spatial_pre_sigmoid, spatial, channel, probabilities = capture_attention(
        model, image, device
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "results"
        / "attention_visualizations"
        / f"{data_metadata['event_id']}_window_{args.window_index:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    display_metadata = render(
        image,
        spatial,
        AF_CHANNELS.index(args.display_channel),
        output_dir,
    )
    attention_weight = model.attention.conv_spatial.weight.detach().cpu()
    np.savez_compressed(
        output_dir / "attention_tensors.npz",
        spatial_attention_pre_sigmoid=spatial_pre_sigmoid,
        spatial_attention=spatial,
        latent_channel_attention=channel,
        fire_probability=probabilities,
    )
    with (output_dir / "latent_channel_attention.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frame_index"]
            + [f"feature_channel_{index}" for index in range(channel.shape[1])]
        )
        for frame_index, values in enumerate(channel, 1):
            writer.writerow([frame_index, *values.tolist()])
    write_json(
        output_dir / "attention_metadata.json",
        {
            "data": data_metadata,
            "model": model_metadata,
            "method": (
                "pre-sigmoid output of the trained attention.conv_spatial "
                "module captured by a forward hook, plus its exact sigmoid "
                "gate for each of 10 frames"
            ),
            "normalization": "same fixed active-fire means/std as training",
            "attention_statistics": display_metadata,
            "pre_sigmoid_statistics": {
                "min": float(spatial_pre_sigmoid.min()),
                "max": float(spatial_pre_sigmoid.max()),
                "mean": float(spatial_pre_sigmoid.mean()),
                "std": float(spatial_pre_sigmoid.std()),
            },
            "spatial_convolution_weight_statistics": {
                "shape": list(attention_weight.shape),
                "min": float(attention_weight.min()),
                "max": float(attention_weight.max()),
                "mean": float(attention_weight.mean()),
                "positive_fraction": float(
                    (attention_weight > 0).float().mean()
                ),
            },
            "latent_channel_attention_statistics": {
                "min": float(channel.min()),
                "max": float(channel.max()),
                "mean": float(channel.mean()),
                "std": float(channel.std()),
                "fraction_lt_0_10": float((channel < 0.10).mean()),
                "fraction_gt_0_90": float((channel > 0.90).mean()),
            },
            "display_note": (
                "Only the relative overlay is rendered. Post-sigmoid attention "
                "is normalized with the sequence-wide p2-p98 range. Relative "
                "values below 0.9 are fully transparent; values from 0.9 to "
                "1.0 use yellow-to-red colors and alpha increasing linearly "
                "from 0.3 to 1.0. "
                "Transparent relative pixels can still have high absolute "
                "attention. Both pre- and post-sigmoid raw tensors are "
                "preserved for audit."
            ),
            "interpretation_limit": (
                "Attention intensity is a model-internal weight, not a causal "
                "explanation, a probability distribution over pixels, a "
                "segmentation mask, or an independently validated localization "
                "score. Sigmoid CBAM/DCBAM gates are not constrained to be "
                "sparse or to sum to one."
            ),
            "latent_channel_note": (
                "The CSV columns are 96 latent Swin feature channels; they are "
                "not the eight physical input bands. Spatial-gate saturation "
                "does not imply that the channel gate is also unselective."
            ),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
