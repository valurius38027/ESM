#!/usr/bin/env python3
"""Strict deterministic aggregation for SGW-ESM Phase 2 experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

PRESETS = (
    "core_small",
    "core_compute_matched",
    "core_param_matched",
    "sgw",
)
INTERVENTIONS = (
    "intact",
    "no_broadcast",
    "no_workspace_persistence",
    "no_workspace_output",
    "permuted_recipients",
)
SUMMARY_METRICS = (
    "initial_train_nll",
    "initial_holdout_nll",
    "final_train_nll",
    "final_holdout_nll",
    "final_train_accuracy",
    "final_holdout_accuracy",
    "mean_estimated_madds_per_token",
    "mean_active_mechanisms_per_token",
    "mean_writers_per_token",
    "mean_recipients_per_token",
    "first_window_loss",
    "last_window_loss",
    "parameters",
)
BASE_BOOTSTRAP_SEED = 20260715


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seed_count <= 0:
        parser.error("seed range must be non-negative and non-empty")
    if args.bootstrap_replicates <= 0:
        parser.error("bootstrap replicates must be positive")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def expected_seed_set(start: int, count: int) -> set[int]:
    return set(range(start, start + count))


def load_runs(raw_dir: Path, expected_seeds: set[int]) -> list[dict[str, str]]:
    paths = sorted(raw_dir.glob("*_seed*.csv"))
    if not paths:
        raise ValueError(f"no run shards in {raw_dir}")
    rows: list[dict[str, str]] = []
    seen: set[tuple[int, str]] = set()
    for path in paths:
        shard = read_csv(path)
        if len(shard) != 1:
            raise ValueError(f"{path} must contain exactly one run row")
        row = shard[0]
        seed = int(row["seed"])
        preset = row["preset"]
        if seed not in expected_seeds:
            raise ValueError(f"unexpected seed {seed} in {path}")
        if preset not in PRESETS:
            raise ValueError(f"unexpected preset {preset} in {path}")
        key = (seed, preset)
        if key in seen:
            raise ValueError(f"duplicate run row for seed={seed}, preset={preset}")
        seen.add(key)
        rows.append(row)
    expected = {(seed, preset) for seed in expected_seeds for preset in PRESETS}
    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        raise ValueError(f"incomplete run Cartesian product: missing={sorted(missing)}, extra={sorted(extra)}")
    rows.sort(key=lambda row: (int(row["seed"]), PRESETS.index(row["preset"])))
    return rows


def load_causal(causal_dir: Path, expected_seeds: set[int]) -> list[dict[str, str]]:
    paths = sorted(causal_dir.glob("causal_seed*.csv"))
    if not paths:
        raise ValueError(f"no causal shards in {causal_dir}")
    rows: list[dict[str, str]] = []
    seen: set[tuple[int, str]] = set()
    for path in paths:
        shard = read_csv(path)
        for row in shard:
            seed = int(row["seed"])
            intervention = row["intervention"]
            if seed not in expected_seeds:
                raise ValueError(f"unexpected causal seed {seed} in {path}")
            if intervention not in INTERVENTIONS:
                raise ValueError(f"unexpected intervention {intervention} in {path}")
            key = (seed, intervention)
            if key in seen:
                raise ValueError(
                    f"duplicate causal row for seed={seed}, intervention={intervention}"
                )
            seen.add(key)
            rows.append(row)
    expected = {
        (seed, intervention)
        for seed in expected_seeds
        for intervention in INTERVENTIONS
    }
    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        raise ValueError(f"incomplete causal Cartesian product: missing={sorted(missing)}, extra={sorted(extra)}")
    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            INTERVENTIONS.index(row["intervention"]),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    return (
        statistics.fmean(materialized),
        statistics.stdev(materialized) if len(materialized) > 1 else 0.0,
    )


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires data")
    index = int(round(probability * (len(sorted_values) - 1)))
    return sorted_values[index]


def paired_stats(
    differences: list[float], replicates: int, seed_offset: int
) -> dict[str, object]:
    if not differences:
        raise ValueError("paired comparison requires differences")
    mean, sd = mean_sd(differences)
    generator = random.Random(BASE_BOOTSTRAP_SEED + seed_offset)
    count = len(differences)
    bootstrap = []
    for _ in range(replicates):
        bootstrap.append(
            statistics.fmean(differences[generator.randrange(count)] for _ in range(count))
        )
    bootstrap.sort()
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    return {
        "seeds": count,
        "mean": mean,
        "sd": sd,
        "ci_lower": percentile(bootstrap, 0.025),
        "ci_upper": percentile(bootstrap, 0.975),
        "cohen_dz": mean / sd if sd > 0.0 else (math.inf if mean > 0.0 else 0.0),
        "positive_seeds": positive,
        "negative_seeds": negative,
        "ties": count - positive - negative,
    }


def summarize_runs(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["preset"]].append(row)
    summaries: list[dict[str, object]] = []
    for preset in PRESETS:
        result: dict[str, object] = {"preset": preset, "seeds": len(grouped[preset])}
        for metric in SUMMARY_METRICS:
            mean, sd = mean_sd(float(row[metric]) for row in grouped[preset])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
        summaries.append(result)
    return summaries


def compare_runs(
    rows: list[dict[str, str]], replicates: int
) -> list[dict[str, object]]:
    by_seed: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][row["preset"]] = row
    results: list[dict[str, object]] = []
    baselines = ("core_small", "core_compute_matched", "core_param_matched")
    for index, baseline in enumerate(baselines):
        differences = [
            float(by_seed[seed][baseline]["final_holdout_nll"])
            - float(by_seed[seed]["sgw"]["final_holdout_nll"])
            for seed in sorted(by_seed)
        ]
        stats = paired_stats(differences, replicates, index)
        results.append(
            {
                "comparison": f"{baseline}_vs_sgw",
                "seeds": stats["seeds"],
                "mean_baseline_minus_sgw_nll": stats["mean"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
                "sd": stats["sd"],
                "cohen_dz": stats["cohen_dz"],
                "positive_seeds": stats["positive_seeds"],
                "negative_seeds": stats["negative_seeds"],
                "ties": stats["ties"],
            }
        )
    return results


def compare_causal(
    rows: list[dict[str, str]], replicates: int
) -> list[dict[str, object]]:
    by_seed: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][row["intervention"]] = row
    results: list[dict[str, object]] = []
    for index, intervention in enumerate(INTERVENTIONS[1:], start=100):
        nll_differences = [
            float(by_seed[seed][intervention]["holdout_nll"])
            - float(by_seed[seed]["intact"]["holdout_nll"])
            for seed in sorted(by_seed)
        ]
        accuracy_differences = [
            float(by_seed[seed][intervention]["holdout_accuracy"])
            - float(by_seed[seed]["intact"]["holdout_accuracy"])
            for seed in sorted(by_seed)
        ]
        stats = paired_stats(nll_differences, replicates, index)
        results.append(
            {
                "intervention": intervention,
                "seeds": stats["seeds"],
                "mean_nll_increase": stats["mean"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
                "sd": stats["sd"],
                "cohen_dz": stats["cohen_dz"],
                "positive_seeds": stats["positive_seeds"],
                "negative_seeds": stats["negative_seeds"],
                "ties": stats["ties"],
                "mean_accuracy_change": statistics.fmean(accuracy_differences),
            }
        )
    return results


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    seeds = expected_seed_set(args.seed_start, args.seed_count)
    run_rows = load_runs(args.raw_dir, seeds)
    causal_rows = load_causal(args.causal_dir, seeds)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    run_path = output / "phase2_runs.csv"
    summary_path = output / "phase2_summary.csv"
    paired_path = output / "phase2_paired.csv"
    causal_path = output / "phase2_causal.csv"
    causal_comparisons_path = output / "phase2_causal_comparisons.csv"

    write_csv(run_path, run_rows)
    write_csv(summary_path, summarize_runs(run_rows))
    write_csv(paired_path, compare_runs(run_rows, args.bootstrap_replicates))
    write_csv(causal_path, causal_rows)
    write_csv(
        causal_comparisons_path,
        compare_causal(causal_rows, args.bootstrap_replicates),
    )

    data_paths = (
        run_path,
        summary_path,
        paired_path,
        causal_path,
        causal_comparisons_path,
    )
    hashes = {path.name: sha256(path) for path in data_paths}
    manifest = {
        "analysis_seed": BASE_BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "causal_rows": len(causal_rows),
        "file_hashes": hashes,
        "presets": list(PRESETS),
        "run_rows": len(run_rows),
        "seed_count": args.seed_count,
        "seed_start": args.seed_start,
    }
    manifest_path = output / "phase2_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    all_paths = sorted((*data_paths, manifest_path), key=lambda path: path.name)
    checksum_path = output / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in all_paths),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
