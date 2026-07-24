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
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.splits import assert_disjoint, load_event_ids  # noqa: E402
from training_protocol import VALIDATION_THRESHOLDS, validate_protocol  # noqa: E402

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _resolve_env(value: str) -> str:
    def _replace(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, match.group(0))

    return _ENV_PATTERN.sub(_replace, value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_path(value: str, field: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Config field {field!r} must be a path string.")
    value = _resolve_env(value)
    unresolved = _ENV_PATTERN.search(value)
    if unresolved:
        name = unresolved.group(1) or unresolved.group(2)
        raise ValueError(f"Environment variable {name!r} required by config field {field!r} is not set.")
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
    unknown = sorted(
        {
            row["event_id"]
            for row in rows
            if not row["event_id"].endswith("_aggregate")
        }
        - declared
    )
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


def validate_runtime_inputs(config: dict) -> dict[str, object]:
    """Resolve and verify every file required before creating a run directory."""
    data = config["data"]
    model = config["model"]
    training = config["training"]
    experiment = config["experiment"]

    data_root = configured_path(data.get("preprocessed_root", ""), "data.preprocessed_root")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Preprocessed data root does not exist: {data_root}")

    mode = "af" if experiment["task"] == "active_fire" else experiment["task"]
    sequence_length = int(training["sequence_length"])
    interval = int(training["interval"])
    dataset_files = {
        "train_images": data_root
        / "dataset_train"
        / f"{mode}_train_img_seqtoseq_alll_{sequence_length}i_{interval}.npy",
        "train_labels": data_root
        / "dataset_train"
        / f"{mode}_train_label_seqtoseq_alll_{sequence_length}i_{interval}.npy",
        "validation_images": data_root
        / "dataset_val"
        / f"{mode}_val_img_seqtoseq_alll_{sequence_length}i_{interval}.npy",
        "validation_labels": data_root
        / "dataset_val"
        / f"{mode}_val_label_seqtoseq_alll_{sequence_length}i_{interval}.npy",
    }
    missing = [str(path) for path in dataset_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required preprocessed arrays are missing: " + ", ".join(missing))

    pretrained_raw = model.get("pretrained_weights", "")
    random_initialization = (
        pretrained_raw is None
        or (
            isinstance(pretrained_raw, str)
            and pretrained_raw.strip().lower() in ("", "none", "null")
        )
    )
    if random_initialization:
        pretrained = None
        pretrained_sha256 = None
    else:
        pretrained = configured_path(pretrained_raw, "model.pretrained_weights")
        if not pretrained.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint does not exist: {pretrained}")
        pretrained_sha256 = sha256(pretrained)

    return {
        "data_root": str(data_root),
        "dataset_files": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in dataset_files.items()
        },
        "pretrained": {
            "mode": "random" if random_initialization else "checkpoint",
            "path": None if pretrained is None else str(pretrained),
            "sha256": pretrained_sha256,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--check", action="store_true", help="Validate only; do not create a run or start training.")
    parser.add_argument("--execute", action="store_true", help="Start the legacy model loop after preflight succeeds.")
    parser.add_argument("--override-attention", default=None, help="Override model.attention in config.")
    parser.add_argument("--override-scheduler", default=None, help="Override training.scheduler in config.")
    parser.add_argument("--override-loss-type", default=None, help="Override loss.name in config.")
    parser.add_argument("--override-pretrained", default=None, help="Override model.pretrained_weights in config.")
    parser.add_argument("--override-output-root", default=None, help="Override experiment.output_root in config.")
    parser.add_argument("--override-tversky-alpha", type=float, default=None, help="Override loss.alpha in config.")
    parser.add_argument("--override-tversky-beta", type=float, default=None, help="Override loss.beta in config.")
    parser.add_argument("--no-copy-paste", action="store_true", help="Disable copy-paste augmentation (for Model B/C/D).")
    args = parser.parse_args()
    if args.check == args.execute:
        parser.error("Choose exactly one of --check or --execute.")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)

    if args.override_attention is not None:
        config.setdefault("model", {})["attention"] = args.override_attention
    if args.override_scheduler is not None:
        config.setdefault("training", {})["scheduler"] = args.override_scheduler
    if args.override_loss_type is not None:
        config.setdefault("loss", {})["name"] = args.override_loss_type
    if args.override_pretrained is not None:
        config.setdefault("model", {})["pretrained_weights"] = args.override_pretrained
    if args.override_output_root is not None:
        config.setdefault("experiment", {})["output_root"] = args.override_output_root
    if args.override_tversky_alpha is not None:
        config.setdefault("loss", {})["alpha"] = args.override_tversky_alpha
    if args.override_tversky_beta is not None:
        config.setdefault("loss", {})["beta"] = args.override_tversky_beta

    experiment = config.get("experiment", {})
    seeds = experiment.get("seeds", [])
    if args.seed not in seeds:
        raise ValueError(f"Seed {args.seed} is not declared in {config_path}: {seeds}")
    training = config.get("training", {})
    protocol = validate_protocol(
        checkpoint_after_epoch=int(training.get("checkpoint_after_epoch", 50)),
        early_stopping_after_epoch=int(
            training.get("early_stopping_after_epoch", 100)
        ),
        thresholds=training.get(
            "validation_thresholds", list(VALIDATION_THRESHOLDS)
        ),
    )
    if int(training["max_epochs"]) <= protocol["checkpoint_after_epoch"]:
        raise ValueError(
            "training.max_epochs must exceed training.checkpoint_after_epoch."
        )
    validation = validate(config)
    runtime_inputs = validate_runtime_inputs(config)
    print(
        json.dumps(
            {
                "config": str(config_path),
                "seed": args.seed,
                "training_protocol": protocol,
                **validation,
                **runtime_inputs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.check:
        return

    data = config["data"]
    model = config["model"]
    loss = config["loss"]
    training = config["training"]
    wandb_settings = training.get("wandb", {})
    wandb_mode = wandb_settings.get("mode", training.get("wandb_mode", "disabled"))
    wandb_project = wandb_settings.get("project", "swinfire_jei_resubmission_v2")
    wandb_entity = wandb_settings.get("entity")
    require_wandb_final_metrics = bool(wandb_settings.get(
        "required_for_final_metrics", training.get("require_wandb_final_metrics", False)
    ))
    if require_wandb_final_metrics and wandb_mode != "online":
        raise ValueError(
            "Final reported metrics require training.wandb.mode: online; "
            "offline or disabled runs are not admissible."
        )
    data_root = Path(runtime_inputs["data_root"])
    pretrained_record = runtime_inputs["pretrained"]
    pretrained = (
        None
        if pretrained_record["mode"] == "random"
        else Path(pretrained_record["path"])
    )
    run_dir = PROJECT_ROOT / experiment["output_root"] / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "config": str(config_path), "config_sha256": sha256(config_path), "seed": args.seed,
        "data_root": str(data_root),
        "dataset_files": runtime_inputs["dataset_files"],
        "pretrained": pretrained_record,
        "training_protocol": protocol,
        "wandb": {
            "mode": wandb_mode,
            "project": wandb_project,
            "entity": wandb_entity,
            "required_for_final_metrics": require_wandb_final_metrics,
        },
        **validation,
    }
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        "--checkpoint-after-epoch", str(protocol["checkpoint_after_epoch"]),
        "--early-stopping-after-epoch", str(protocol["early_stopping_after_epoch"]),
        "--validation-thresholds", *[
            str(value) for value in protocol["validation_thresholds"]
        ],
        "-scheduler", training["scheduler"], "--data-root", str(data_root),
        "--output-dir", str(run_dir),
        "--wandb-mode", wandb_mode, "--wandb-project", str(wandb_project),
        "--run-manifest", str(run_manifest_path),
        "--loss-type", loss["name"], "-tversky_alpha", str(loss.get("alpha", 0.5)),
        "-tversky_beta", str(loss.get("beta", 0.5)), "-focal_gamma", str(loss.get("gamma", 3.0)),
    ]
    if pretrained is None:
        command.append("--allow-random-init")
    else:
        command.extend(["--pretrained-path", str(pretrained)])
    if args.no_copy_paste:
        command.append("--no-copy-paste")
    if wandb_entity:
        command.extend(["--wandb-entity", str(wandb_entity)])
    if require_wandb_final_metrics:
        command.append("--wandb-require-final-metrics")
    (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
