#!/usr/bin/env python3
"""Create convergence-curve data and figures from immutable epoch JSONL logs.

This script does not train or evaluate a model and never infers missing epochs.
Every plotted point is read from ``epoch_metrics.jsonl`` written by the trainer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED = {"record_type", "model", "attention", "loss_type", "seed", "epoch", "train_loss", "val_loss", "val_f1_score"}


def load_epoch_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("epoch_metrics.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} misses {sorted(missing)}")
            if row["record_type"] != "training_epoch":
                raise ValueError(f"{path}:{line_number} is not a training_epoch record")
            row["source_log"] = str(path)
            row["seed"] = int(row["seed"])
            row["epoch"] = int(row["epoch"])
            for key in ("train_loss", "val_loss", "val_f1_score"):
                row[key] = float(row[key])
            records.append(row)
    if not records:
        raise ValueError(f"No epoch_metrics.jsonl logs found below {input_dir}")
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/training_runs"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    records = load_epoch_records(args.input)
    records.sort(key=lambda row: (row["model"], row["attention"], row["loss_type"], row["seed"], row["epoch"]))
    write_csv(args.output / "figure_data" / "training_epoch_metrics.csv", records)

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["attention"], row["loss_type"], row["seed"])].append(row)

    for metric, ylabel in (("train_loss", "Training loss"), ("val_loss", "Validation loss"), ("val_f1_score", "Validation F1")):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for (model, attention, loss_type, seed), rows in sorted(grouped.items()):
            rows.sort(key=lambda row: row["epoch"])
            axis.plot(
                [row["epoch"] for row in rows],
                [row[metric] for row in rows],
                label=f"{model} | {attention} | {loss_type} | seed {seed}",
            )
        axis.set(xlabel="Epoch", ylabel=ylabel, title=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        output_path = args.output / "figures" / f"{metric}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200)
        plt.close(figure)

    print(f"Wrote {len(records)} epoch records and 3 figures from {len(grouped)} independent runs.")


if __name__ == "__main__":
    main()
