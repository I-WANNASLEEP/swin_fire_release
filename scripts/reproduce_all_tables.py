#!/usr/bin/env python3
"""Generate statistical tables and figure-data files only from raw metric records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_COLUMNS = {
    "experiment", "model", "attention", "seed", "split", "event_id",
    "f1", "iou", "precision", "recall", "checkpoint_sha256", "dataset_sha256",
    "sample_manifest_sha256",
}
METRICS = ("f1", "iou", "precision", "recall")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, str]] = []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.extend(payload if isinstance(payload, list) else [payload])
        elif path.suffix == ".jsonl":
            records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                records.extend(csv.DictReader(handle))
        else:
            continue
        provenance.append({"path": str(path), "sha256": file_sha256(path)})
    if not records:
        raise ValueError(f"No JSON, JSONL, or CSV raw metric records found under {input_dir}.")
    for record in records:
        missing = REQUIRED_COLUMNS - set(record)
        if missing:
            raise ValueError(f"Raw metric record is missing required fields {sorted(missing)}: {record}")
        record["seed"] = int(record["seed"])
        for metric in METRICS:
            record[metric] = float(record[metric])
    return records, provenance


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values)


def event_bootstrap_ci(rows: list[dict[str, Any]], metric: str, samples: int, seed: int) -> tuple[float, float]:
    by_event: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_event[row["event_id"]].append(row[metric])
    event_means = [mean(values) for _, values in sorted(by_event.items())]
    if len(event_means) < 2:
        return (event_means[0], event_means[0])
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(event_means, k=len(event_means))) for _ in range(samples))
    return draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def wandb_key_fragment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "unspecified"


def log_final_summary_to_wandb(
    summary: list[dict[str, Any]], records: list[dict[str, Any]], summary_path: Path,
    provenance_path: Path, args: argparse.Namespace,
) -> None:
    """Record the exact paper-level scalar summaries in a dedicated online W&B run."""
    if args.wandb_mode == "disabled":
        if args.require_wandb:
            raise ValueError("--require-wandb requires --wandb-mode online.")
        return
    if args.require_wandb and args.wandb_mode != "online":
        raise ValueError("Final paper summaries require online W&B tracking, not offline mode.")
    if args.require_wandb:
        missing_wandb = [
            row["event_id"] for row in records if not row.get("wandb_run_id") or not row.get("wandb_run_url")
        ]
        if missing_wandb:
            raise ValueError(
                "Final paper summaries require every raw event record to identify its W&B run; "
                f"missing examples: {missing_wandb[:5]}"
            )
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError("W&B is required for this final-summary run but is not installed.") from exc

    if args.wandb_mode == "online":
        wandb.login()
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        job_type="paper_final_summary",
        name=args.wandb_report_name,
        config={
            "split": args.split,
            "minimum_independent_seeds": args.min_seeds,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "event",
            "raw_event_record_count": len(records),
            "summary_csv_sha256": file_sha256(summary_path),
            "provenance_sha256": file_sha256(provenance_path),
        },
    )
    try:
        for row in summary:
            group = "__".join(wandb_key_fragment(str(row[field])) for field in ("experiment", "model", "attention"))
            payload: dict[str, Any] = {
                "final_summary/group": group,
                "final_summary/experiment": row["experiment"],
                "final_summary/model": row["model"],
                "final_summary/attention": row["attention"],
                "final_summary/n_seeds": row["n_seeds"],
                "final_summary/n_events": row["n_events"],
            }
            for metric in METRICS:
                for suffix in ("mean", "std", "event_bootstrap_ci95_low", "event_bootstrap_ci95_high"):
                    value = row[f"{metric}_{suffix}"]
                    payload[f"final_summary/{metric}_{suffix}"] = value
                    payload[f"final_summary_by_group/{group}/{metric}_{suffix}"] = value
            wandb.log(payload)
        columns = list(summary[0]) if summary else []
        if columns:
            table = wandb.Table(columns=columns, data=[[row[column] for column in columns] for row in summary])
            wandb.log({"final_summary/model_metrics_table": table})
        run.summary.update({
            "final_summary/summary_rows": len(summary),
            "final_summary/raw_event_records": len(records),
            "final_summary/summary_csv_sha256": file_sha256(summary_path),
            "final_summary/provenance_sha256": file_sha256(provenance_path),
        })
    finally:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/raw_metrics"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-project", default="ts_satfire_jei_resubmission")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-report-name", default="final-test-summary")
    parser.add_argument("--require-wandb", action="store_true",
                        help="Require every paper-summary scalar to be uploaded to online W&B.")
    args = parser.parse_args()

    records, provenance = load_records(args.input)
    records = [record for record in records if record["split"] == args.split]
    if not records:
        raise ValueError(f"No records for split {args.split!r}.")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["experiment"], record["model"], record["attention"])].append(record)

    summary: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        seeds = sorted({row["seed"] for row in rows})
        if len(seeds) < args.min_seeds:
            raise ValueError(f"{key} has {len(seeds)} seeds; at least {args.min_seeds} are required.")
        row: dict[str, Any] = {
            "experiment": key[0], "model": key[1], "attention": key[2],
            "split": args.split, "n_seeds": len(seeds), "n_events": len({r['event_id'] for r in rows}),
        }
        for metric in METRICS:
            per_seed = [mean(r[metric] for r in rows if r["seed"] == seed) for seed in seeds]
            row[f"{metric}_mean"] = mean(per_seed)
            row[f"{metric}_std"] = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
            lo, hi = event_bootstrap_ci(rows, metric, args.bootstrap_samples, args.bootstrap_seed)
            row[f"{metric}_event_bootstrap_ci95_low"] = lo
            row[f"{metric}_event_bootstrap_ci95_high"] = hi
        summary.append(row)

    summary_path = args.output / "summary_tables" / f"{args.split}_model_summary.csv"
    write_csv(summary_path, summary)
    write_csv(args.output / "figure_data" / f"{args.split}_raw_seed_event_metrics.csv", records)
    write_csv(args.output / "figure_data" / f"{args.split}_attention_ablation_summary.csv", [
        row for row in summary if row["attention"] != "not_applicable"
    ])
    provenance_path = args.output / "summary_tables" / f"{args.split}_provenance.json"
    provenance_path.write_text(json.dumps({
        "raw_metric_inputs": provenance,
        "split": args.split,
        "minimum_independent_seeds": args.min_seeds,
        "bootstrap": {"unit": "event", "samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_final_summary_to_wandb(summary, records, summary_path, provenance_path, args)
    print(f"Generated {len(summary)} summary row(s) from {len(records)} raw records.")


if __name__ == "__main__":
    main()
