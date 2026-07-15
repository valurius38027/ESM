#!/usr/bin/env python3
"""Consolidate deterministic SGW-ESM CSV shards using only stdlib."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "initial_holdout_nll",
    "final_holdout_nll",
    "final_holdout_accuracy",
    "final_train_nll",
    "mean_estimated_madds_per_token",
    "parameters",
    "first_window_loss",
    "last_window_loss",
)


def read_rows(raw_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(raw_dir.glob("*_seed*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            shard = list(csv.DictReader(stream))
        if len(shard) != 1:
            raise ValueError(f"{path} must contain exactly one data row")
        rows.extend(shard)
    if not rows:
        raise ValueError(f"no CSV shards found in {raw_dir}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    paired: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        mode = row["mode"]
        seed = int(row["seed"])
        if mode not in {"core_only", "sgw"}:
            raise ValueError(f"unexpected mode {mode}")
        if mode in paired[seed]:
            raise ValueError(f"duplicate mode {mode} for seed {seed}")
        grouped[mode].append(row)
        paired[seed][mode] = row

    if set(grouped) != {"core_only", "sgw"}:
        raise ValueError("both core_only and sgw modes are required")
    seed_sets = {mode: {int(row["seed"]) for row in mode_rows}
                 for mode, mode_rows in grouped.items()}
    if seed_sets["core_only"] != seed_sets["sgw"]:
        raise ValueError("core_only and sgw seed sets must match")

    summary_rows: list[dict[str, object]] = []
    for mode in ("core_only", "sgw"):
        mode_rows = grouped[mode]
        result: dict[str, object] = {"mode": mode, "seeds": len(mode_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in mode_rows]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(result)

    paired_rows: list[dict[str, object]] = []
    for seed in sorted(paired):
        pair = paired[seed]
        core_nll = float(pair["core_only"]["final_holdout_nll"])
        sgw_nll = float(pair["sgw"]["final_holdout_nll"])
        paired_rows.append({
            "seed": seed,
            "core_only_final_holdout_nll": core_nll,
            "sgw_final_holdout_nll": sgw_nll,
            "core_minus_sgw_nll": core_nll - sgw_nll,
            "core_only_final_holdout_accuracy": float(
                pair["core_only"]["final_holdout_accuracy"]),
            "sgw_final_holdout_accuracy": float(
                pair["sgw"]["final_holdout_accuracy"]),
        })

    differences = [float(row["core_minus_sgw_nll"]) for row in paired_rows]
    mean = statistics.fmean(differences)
    sd = statistics.stdev(differences) if len(differences) > 1 else 0.0
    se = sd / math.sqrt(len(differences)) if differences else 0.0
    paired_rows.append({
        "seed": "mean",
        "core_only_final_holdout_nll": "",
        "sgw_final_holdout_nll": "",
        "core_minus_sgw_nll": mean,
        "core_only_final_holdout_accuracy": "",
        "sgw_final_holdout_accuracy": "",
    })
    paired_rows.append({
        "seed": "sd",
        "core_only_final_holdout_nll": "",
        "sgw_final_holdout_nll": "",
        "core_minus_sgw_nll": sd,
        "core_only_final_holdout_accuracy": "",
        "sgw_final_holdout_accuracy": "",
    })
    paired_rows.append({
        "seed": "se",
        "core_only_final_holdout_nll": "",
        "sgw_final_holdout_nll": "",
        "core_minus_sgw_nll": se,
        "core_only_final_holdout_accuracy": "",
        "sgw_final_holdout_accuracy": "",
    })
    return summary_rows, paired_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = read_rows(args.raw_dir)
    rows.sort(key=lambda row: (int(row["seed"]), row["mode"]))
    summary_rows, paired_rows = summarize(rows)
    write_csv(args.output_dir / "formal_runs.csv", rows)
    write_csv(args.output_dir / "formal_summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_results.csv", paired_rows)


if __name__ == "__main__":
    main()
