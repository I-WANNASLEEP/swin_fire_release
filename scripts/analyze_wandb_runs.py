#!/usr/bin/env python3
"""Export a protocol-aware audit of every run in one W&B project."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import wandb

METRIC_KEYS = [
    "_step",
    "epoch",
    "train_loss",
    "train_tversky_loss",
    "train_focal_loss",
    "train_ce_loss",
    "val_loss",
    "validation/f1_at_selected_threshold",
    "validation/iou_at_selected_threshold",
    "validation/precision_at_selected_threshold",
    "validation/recall_at_selected_threshold",
    "validation/selected_threshold",
    "validation/f1_at_fixed_0_5",
    "learning_rate/unified_start",
    "learning_rate/unified_end",
    "training/optimization_steps",
    "training/processed_samples",
]


def nested(config: dict, *keys: str, default=None):
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def plain(value: Any):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.item()
    except (AttributeError, ValueError):
        return str(value)


def history_rows(run) -> list[dict]:
    rows = []
    try:
        # Do not pass the full key list here. Older runs legitimately lack some
        # newer provenance fields, and W&B otherwise intersects the requested
        # schema and can return no rows at all.
        source = run.scan_history(page_size=500)
        for row in source:
            if row.get("epoch") is not None:
                rows.append(
                    {key: plain(row.get(key)) for key in METRIC_KEYS}
                )
    except Exception:
        sampled = run.history(pandas=False, samples=10000)
        rows = [
            {key: plain(row.get(key)) for key in METRIC_KEYS}
            for row in sampled
            if row.get("epoch") is not None
        ]
    deduplicated = {}
    for row in rows:
        if row["epoch"] is not None:
            deduplicated[int(row["epoch"])] = row
    return [deduplicated[key] for key in sorted(deduplicated)]


def metadata_commit(run) -> str | None:
    try:
        return nested(run.metadata or {}, "git", "commit")
    except Exception:
        return None


def fingerprint(record: dict) -> str:
    controls = {
        key: record.get(key)
        for key in (
            "git_commit",
            "model",
            "attention",
            "seed_independent_initialization",
            "pretrained_sha256",
            "alpha",
            "beta",
            "focal_gamma",
            "loss_type",
            "batch_size",
            "learning_rate",
            "planned_epochs",
            "scheduler",
            "checkpoint_after_epoch",
            "early_stopping_after_epoch",
            "validation_thresholds",
            "lr_trace_sha256",
        )
    }
    return hashlib.sha256(
        json.dumps(controls, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else None
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default="15145202826-1/swinfire_jei_resubmission_v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    api = wandb.Api(timeout=90)
    run_records = []
    all_epoch_rows = []
    for run in api.runs(args.project):
        config = dict(run.config)
        rows = history_rows(run)
        for row in rows:
            row.update({"run_id": run.id, "run_name": run.name})
            all_epoch_rows.append(row)
        metric_rows = [
            row
            for row in rows
            if row.get("validation/f1_at_selected_threshold") is not None
        ]
        final = metric_rows[-1] if metric_rows else {}
        best = (
            max(
                metric_rows,
                key=lambda row: row[
                    "validation/f1_at_selected_threshold"
                ],
            )
            if metric_rows
            else {}
        )
        lr_trace = [
            [
                row.get("epoch"),
                row.get("learning_rate/unified_start"),
                row.get("learning_rate/unified_end"),
            ]
            for row in metric_rows
        ]
        pretrained = config.get("pretrained") or {}
        initialization = config.get("initialization") or {}
        training_protocol = config.get("training_protocol") or {}
        losses = [
            row["train_loss"]
            for row in rows
            if row.get("train_loss") is not None
        ]
        tversky_losses = [
            row["train_tversky_loss"]
            for row in rows
            if row.get("train_tversky_loss") is not None
        ]
        optimization_steps = [
            row["training/optimization_steps"]
            for row in rows
            if row.get("training/optimization_steps") is not None
        ]
        planned_epochs = config.get("epochs")
        last_epoch = final.get("epoch")
        complete = (
            run.state == "finished"
            and planned_epochs is not None
            and last_epoch is not None
            and int(last_epoch) == int(planned_epochs)
        )
        record = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "git_commit": metadata_commit(run),
            "seed": config.get("seed"),
            "model": config.get("model"),
            "attention": config.get("attention_version"),
            "loss_type": config.get("loss_type"),
            "alpha": config.get("tversky_alpha"),
            "beta": config.get("tversky_beta"),
            "focal_gamma": config.get("focal_gamma"),
            "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "planned_epochs": planned_epochs,
            "scheduler": config.get("scheduler"),
            "pretrained_loaded": pretrained.get("loaded"),
            "pretrained_sha256": pretrained.get("sha256"),
            "pretrained_layers": pretrained.get("loaded_layers"),
            "seed_independent_initialization": initialization.get("strategy"),
            "checkpoint_after_epoch": training_protocol.get(
                "checkpoint_after_epoch"
            ),
            "early_stopping_after_epoch": training_protocol.get(
                "early_stopping_after_epoch"
            ),
            "validation_thresholds": json.dumps(
                training_protocol.get("validation_thresholds")
            ),
            "run_manifest_path": config.get("run_manifest_path"),
            "last_epoch": last_epoch,
            "best_epoch": best.get("epoch"),
            "best_validation_f1": best.get(
                "validation/f1_at_selected_threshold"
            ),
            "best_validation_iou": best.get(
                "validation/iou_at_selected_threshold"
            ),
            "best_validation_precision": best.get(
                "validation/precision_at_selected_threshold"
            ),
            "best_validation_recall": best.get(
                "validation/recall_at_selected_threshold"
            ),
            "best_validation_threshold": best.get(
                "validation/selected_threshold"
            ),
            "final_validation_f1": final.get(
                "validation/f1_at_selected_threshold"
            ),
            "complete_as_planned": complete,
            "negative_loss_detected": any(value < 0 for value in losses),
            "negative_tversky_detected": any(
                value < 0 for value in tversky_losses
            ),
            "zero_optimization_epoch_detected": any(
                value <= 0 for value in optimization_steps
            ),
            "lr_trace_sha256": hashlib.sha256(
                json.dumps(lr_trace, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
            "immutable_data_hash_recorded": bool(
                nested(config, "data_provenance", "dataset_sha256")
            ),
        }
        record["protocol_fingerprint"] = fingerprint(record)
        run_records.append(record)

    grouped = defaultdict(list)
    for record in run_records:
        grouped[record["protocol_fingerprint"]].append(record)
    group_records = []
    for group_id, records in sorted(grouped.items()):
        valid = [
            record
            for record in records
            if record["complete_as_planned"]
            and not record["negative_loss_detected"]
            and not record["negative_tversky_detected"]
            and not record["zero_optimization_epoch_detected"]
        ]
        f1_values = [
            record["best_validation_f1"]
            for record in valid
            if record["best_validation_f1"] is not None
        ]
        mean_f1, sd_f1 = mean_sd(f1_values)
        unique_seeds = len({record["seed"] for record in valid})
        group_records.append(
            {
                "protocol_fingerprint": group_id,
                "run_count": len(records),
                "valid_complete_run_count": len(valid),
                "unique_seed_count": unique_seeds,
                "run_ids": " ".join(record["run_id"] for record in records),
                "mean_best_validation_f1": mean_f1,
                "sd_best_validation_f1": sd_f1,
                "statistically_sufficient": unique_seeds >= 3,
                "paper_ready": (
                    unique_seeds >= 3
                    and all(
                        record["immutable_data_hash_recorded"]
                        for record in valid
                    )
                ),
            }
        )

    write_csv(output_dir / "runs.csv", run_records)
    write_csv(output_dir / "epoch_metrics.csv", all_epoch_rows)
    write_csv(output_dir / "protocol_groups.csv", group_records)
    counts = Counter(record["state"] for record in run_records)
    sufficient = [
        record
        for record in group_records
        if record["statistically_sufficient"]
    ]
    paper_ready = [record for record in group_records if record["paper_ready"]]
    summary = {
        "project": args.project,
        "run_count": len(run_records),
        "state_counts": dict(counts),
        "negative_loss_runs": [
            record["run_id"]
            for record in run_records
            if record["negative_loss_detected"]
            or record["negative_tversky_detected"]
        ],
        "zero_optimization_runs": [
            record["run_id"]
            for record in run_records
            if record["zero_optimization_epoch_detected"]
        ],
        "statistically_sufficient_protocol_groups": [
            record["protocol_fingerprint"] for record in sufficient
        ],
        "paper_ready_protocol_groups": [
            record["protocol_fingerprint"] for record in paper_ready
        ],
        "paper_ready_definition": (
            "at least 3 complete independent seeds under one exact protocol, "
            "finite nonnegative losses, positive optimizer steps, and an "
            "immutable dataset hash recorded in W&B"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
