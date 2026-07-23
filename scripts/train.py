#!/usr/bin/env python3
"""Configuration-checked launcher for a single independent training seed.

It validates the fixed split and sample manifest before delegating the model loop
to the maintained legacy backend.  One invocation equals one independent seed;
do not convert epochs into repetitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.splits import assert_disjoint, load_event_ids  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_path(value: str, field: str) -> Path:
    if not value or value.startswith("path/to/"):
        raise ValueError(f"Set a real path for config field {field!r}; placeholders cannot launch a run.")
    return Path(value).expanduser().resolve()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping.")
    return config


def validate(config: dict) -> dict[str, object]:
    data = config.get("data", {})
    splits = data.get("splits", {})
    required = {"train", "validation", "test"}
    if required - set(splits):
        raise ValueError(f"data.splits must define {sorted(required)}")
    loaded = {name: load_event_ids(PROJECT_ROOT / relative) for name, relative in splits.items()}
    assert_disjoint(loaded)

    manifest = configured_path(data.get("required_sample_manifest", ""), "data.required_sample_manifest")
    if not manifest.is_file():
        raise FileNotFoundError(f"Sample manifest does not exist: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"event_id", "split"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Sample manifest misses required columns: {sorted(missing)}")
        rows = list(reader)
    declared = {event_id for ids in loaded.values() for event_id in ids}
    unknown = sorted({row["event_id"] for row in rows} - declared)
    if unknown:
        raise ValueError(f"Sample manifest contains events outside the locked split: {unknown[:5]}")
    if not rows:
        raise ValueError("Sample manifest has no generated windows.")
    return {
        "sample_manifest": str(manifest),
        "sample_manifest_sha256": sha256(manifest),
        "split_counts": {name: len(ids) for name, ids in loaded.items()},
        "split_sha256": {name: sha256(PROJECT_ROOT / relative) for name, relative in splits.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--check", action="store_true", help="Validate only; do not create a run or start training.")
    parser.add_argument("--execute", action="store_true", help="Start the legacy model loop after preflight succeeds.")
    args = parser.parse_args()
    if args.check == args.execute:
        parser.error("Choose exactly one of --check or --execute.")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    experiment = config.get("experiment", {})
    seeds = experiment.get("seeds", [])
    if args.seed not in seeds:
        raise ValueError(f"Seed {args.seed} is not declared in {config_path}: {seeds}")
    validation = validate(config)
    print(json.dumps({"config": str(config_path), "seed": args.seed, **validation}, indent=2, sort_keys=True))
    if args.check:
        return

    data = config["data"]
    model = config["model"]
    loss = config["loss"]
    training = config["training"]
    data_root = configured_path(data.get("preprocessed_root", ""), "data.preprocessed_root")
    pretrained = configured_path(model.get("pretrained_weights", ""), "model.pretrained_weights")
    run_dir = PROJECT_ROOT / experiment["output_root"] / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "config": str(config_path), "config_sha256": sha256(config_path), "seed": args.seed,
        "data_root": str(data_root), "pretrained_weights": str(pretrained), **validation,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Legacy seed convention is run + 41.  The run index therefore remains
    # explicit and is recorded alongside the requested seed.
    run_index = args.seed - 41
    legacy_mode = "af" if experiment["task"] == "active_fire" else experiment["task"]
    command = [
        sys.executable, str(PROJECT_ROOT / "train_models_spatial_temp.py"),
        "-m", model["name"], "-mode", legacy_mode, "-b", str(training["batch_size"]),
        "-r", str(run_index), "-lr", str(training["learning_rate"]), "-av", model["attention"],
        "-nh", str(training["num_heads"]), "-ed", str(model["feature_size"]),
        "-nc", str(model["input_channels"]), "-ts", str(training["sequence_length"]),
        "-it", str(training["interval"]), "--max-epochs", str(training["max_epochs"]),
        "-patience", str(training["patience"]), "-grad_clip", str(training["grad_clip"]),
        "-scheduler", training["scheduler"], "--data-root", str(data_root),
        "--pretrained-path", str(pretrained), "--output-dir", str(run_dir),
        "--wandb-mode", training.get("wandb_mode", "disabled"),
        "--loss-type", loss["name"], "-tversky_alpha", str(loss.get("alpha", 0.5)),
        "-tversky_beta", str(loss.get("beta", 0.5)), "-focal_gamma", str(loss.get("gamma", 3.0)),
    ]
    (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
