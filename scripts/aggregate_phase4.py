#!/usr/bin/env python3
"""Strict deterministic aggregation for SGW-ESM Phase 4 experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

CONDITIONS = (
    "core_full_content",
    "core_content_blind",
    "mediation_final_only",
    "mediation_aux_annealed",
)
INTERVENTIONS = (
    "intact",
    "no_broadcast",
    "no_workspace_persistence",
    "no_workspace_output",
    "no_spine_workspace",
    "no_mechanism_output",
    "workspace_disconnected",
    "no_workspace_writes",
    "zero_reader_inbox",
    "permuted_recipients",
)
ROLES = ("entity", "value", "filler", "query_marker", "query_entity")
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
    "first_primary_window_loss",
    "last_primary_window_loss",
    "last_workspace_aux_window_loss",
    "anneal_boundary_primary_window_loss",
    "final_workspace_aux_weight",
    "final_weighted_workspace_aux_window_loss",
    "parameters",
)
COMPARISONS = (
    ("blind_vs_full", "core_content_blind", "core_full_content"),
    ("mediation_final_vs_full", "core_full_content", "mediation_final_only"),
    ("mediation_aux_vs_final", "mediation_final_only", "mediation_aux_annealed"),
    ("mediation_aux_vs_full", "core_full_content", "mediation_aux_annealed"),
    ("mediation_aux_vs_blind", "core_content_blind", "mediation_aux_annealed"),
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
        condition = row["condition"]
        if seed not in expected_seeds:
            raise ValueError(f"unexpected seed {seed} in {path}")
        if condition not in CONDITIONS:
            raise ValueError(f"unexpected condition {condition} in {path}")
        key = (seed, condition)
        if key in seen:
            raise ValueError(
                f"duplicate run row for seed={seed}, condition={condition}"
            )
        seen.add(key)
        parse_role_matrix(row["role_mechanism_load"])
        rows.append(row)
    expected = {
        (seed, condition) for seed in expected_seeds for condition in CONDITIONS
    }
    if seen != expected:
        raise ValueError(
            "incomplete run Cartesian product: "
            f"missing={sorted(expected - seen)}, extra={sorted(seen - expected)}"
        )
    rows.sort(
        key=lambda row: (int(row["seed"]), CONDITIONS.index(row["condition"]))
    )
    return rows


def load_causal(causal_dir: Path, expected_seeds: set[int]) -> list[dict[str, str]]:
    paths = sorted(causal_dir.glob("causal_seed*.csv"))
    if not paths:
        raise ValueError(f"no causal shards in {causal_dir}")
    rows: list[dict[str, str]] = []
    seen: set[tuple[int, str]] = set()
    for path in paths:
        for row in read_csv(path):
            seed = int(row["seed"])
            intervention = row["intervention"]
            condition = row["condition"]
            if seed not in expected_seeds:
                raise ValueError(f"unexpected causal seed {seed} in {path}")
            if intervention not in INTERVENTIONS:
                raise ValueError(f"unexpected intervention {intervention} in {path}")
            if condition != "mediation_aux_annealed":
                raise ValueError(f"unexpected causal condition {condition} in {path}")
            key = (seed, intervention)
            if key in seen:
                raise ValueError(
                    f"duplicate causal row for seed={seed}, intervention={intervention}"
                )
            seen.add(key)
            parse_role_matrix(row["role_mechanism_load"])
            rows.append(row)
    expected = {
        (seed, intervention)
        for seed in expected_seeds
        for intervention in INTERVENTIONS
    }
    if seen != expected:
        raise ValueError(
            "incomplete causal Cartesian product: "
            f"missing={sorted(expected - seen)}, extra={sorted(seen - expected)}"
        )
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
    bootstrap = [
        statistics.fmean(
            differences[generator.randrange(count)] for _ in range(count)
        )
        for _ in range(replicates)
    ]
    bootstrap.sort()
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    return {
        "seeds": count,
        "mean": mean,
        "sd": sd,
        "ci_lower": percentile(bootstrap, 0.025),
        "ci_upper": percentile(bootstrap, 0.975),
        "cohen_dz": mean / sd if sd > 0.0 else 0.0,
        "positive_seeds": positive,
        "negative_seeds": negative,
        "ties": count - positive - negative,
    }


def summarize_runs(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    summaries: list[dict[str, object]] = []
    for condition in CONDITIONS:
        result: dict[str, object] = {
            "condition": condition,
            "seeds": len(grouped[condition]),
        }
        for metric in SUMMARY_METRICS:
            mean, sd = mean_sd(float(row[metric]) for row in grouped[condition])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
        summaries.append(result)
    return summaries


def compare_runs(
    rows: list[dict[str, str]], replicates: int
) -> list[dict[str, object]]:
    by_seed: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][row["condition"]] = row
    results: list[dict[str, object]] = []
    for index, (name, baseline, candidate) in enumerate(COMPARISONS):
        differences = [
            float(by_seed[seed][baseline]["final_holdout_nll"])
            - float(by_seed[seed][candidate]["final_holdout_nll"])
            for seed in sorted(by_seed)
        ]
        stats = paired_stats(differences, replicates, index)
        results.append(
            {
                "comparison": name,
                "baseline": baseline,
                "candidate": candidate,
                "seeds": stats["seeds"],
                "mean_baseline_minus_candidate_nll": stats["mean"],
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


def parse_role_matrix(text: str) -> list[list[int]]:
    rows: dict[str, list[int]] = {}
    for encoded in text.split("|"):
        if ":" not in encoded:
            raise ValueError(f"invalid role mechanism load row: {encoded}")
        role, values_text = encoded.split(":", 1)
        if role in rows:
            raise ValueError(f"duplicate role {role}")
        values = [int(value) for value in values_text.split(";")]
        if not values or any(value < 0 for value in values):
            raise ValueError(f"invalid mechanism counts for role {role}")
        rows[role] = values
    if set(rows) != set(ROLES):
        raise ValueError(f"role matrix mismatch: {sorted(rows)}")
    mechanism_counts = {len(values) for values in rows.values()}
    if len(mechanism_counts) != 1:
        raise ValueError("role rows have inconsistent mechanism counts")
    return [rows[role] for role in ROLES]


def entropy(probabilities: Iterable[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def route_statistics(text: str) -> dict[str, object]:
    matrix = parse_role_matrix(text)
    role_count = len(matrix)
    mechanism_count = len(matrix[0])
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return {
            "mechanism_count": mechanism_count,
            "normalized_mechanism_entropy": 0.0,
            "role_mechanism_mutual_information": 0.0,
            "normalized_role_mechanism_mutual_information": 0.0,
            "dominant": [None] * role_count,
        }
    role_totals = [sum(row) for row in matrix]
    mechanism_totals = [
        sum(matrix[role][mechanism] for role in range(role_count))
        for mechanism in range(mechanism_count)
    ]
    role_probabilities = [value / total for value in role_totals]
    mechanism_probabilities = [value / total for value in mechanism_totals]
    mechanism_entropy = entropy(mechanism_probabilities)
    mutual_information = 0.0
    for role in range(role_count):
        for mechanism in range(mechanism_count):
            joint = matrix[role][mechanism] / total
            if joint > 0.0:
                mutual_information += joint * math.log(
                    joint
                    / (role_probabilities[role] * mechanism_probabilities[mechanism])
                )
    mutual_information = max(0.0, mutual_information)
    normalization = min(math.log(role_count), math.log(mechanism_count))
    dominant = []
    for row in matrix:
        if sum(row) == 0:
            dominant.append(None)
        else:
            dominant.append(max(range(mechanism_count), key=lambda index: (row[index], -index)))
    return {
        "mechanism_count": mechanism_count,
        "normalized_mechanism_entropy": (
            mechanism_entropy / math.log(mechanism_count)
            if mechanism_count > 1
            else 0.0
        ),
        "role_mechanism_mutual_information": mutual_information,
        "normalized_role_mechanism_mutual_information": (
            mutual_information / normalization if normalization > 0.0 else 0.0
        ),
        "dominant": dominant,
    }


def routing_outputs(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(route_statistics(row["role_mechanism_load"]))
    summary: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    for condition in CONDITIONS:
        stats = grouped[condition]
        entropy_mean, entropy_sd = mean_sd(
            float(item["normalized_mechanism_entropy"]) for item in stats
        )
        mi_mean, mi_sd = mean_sd(
            float(item["role_mechanism_mutual_information"]) for item in stats
        )
        normalized_mi_mean, normalized_mi_sd = mean_sd(
            float(item["normalized_role_mechanism_mutual_information"])
            for item in stats
        )
        summary.append(
            {
                "condition": condition,
                "seeds": len(stats),
                "normalized_mechanism_entropy_mean": entropy_mean,
                "normalized_mechanism_entropy_sd": entropy_sd,
                "role_mechanism_mutual_information_mean": mi_mean,
                "role_mechanism_mutual_information_sd": mi_sd,
                "normalized_role_mechanism_mutual_information_mean": normalized_mi_mean,
                "normalized_role_mechanism_mutual_information_sd": normalized_mi_sd,
            }
        )
        for role_index, role in enumerate(ROLES):
            values = [item["dominant"][role_index] for item in stats]
            present = [value for value in values if value is not None]
            if present:
                counts = Counter(present)
                mode, mode_count = min(
                    counts.items(), key=lambda item: (-item[1], item[0])
                )
                mode_value: object = mode
                fraction = mode_count / len(stats)
            else:
                mode_value = ""
                mode_count = 0
                fraction = 0.0
            stability.append(
                {
                    "condition": condition,
                    "role": role,
                    "seeds": len(stats),
                    "modal_dominant_mechanism": mode_value,
                    "modal_count": mode_count,
                    "stability_fraction": fraction,
                }
            )
    return summary, stability


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
    routing_rows, routing_stability_rows = routing_outputs(run_rows)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    outputs: list[tuple[Path, list[dict[str, object]] | list[dict[str, str]]]] = [
        (output / "phase4_runs.csv", run_rows),
        (output / "phase4_summary.csv", summarize_runs(run_rows)),
        (output / "phase4_paired.csv", compare_runs(run_rows, args.bootstrap_replicates)),
        (output / "phase4_causal.csv", causal_rows),
        (
            output / "phase4_causal_comparisons.csv",
            compare_causal(causal_rows, args.bootstrap_replicates),
        ),
        (output / "phase4_routing.csv", routing_rows),
        (output / "phase4_routing_stability.csv", routing_stability_rows),
    ]
    for path, rows in outputs:
        write_csv(path, rows)

    data_paths = tuple(path for path, _ in outputs)
    hashes = {path.name: sha256(path) for path in data_paths}
    manifest = {
        "analysis_seed": BASE_BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "causal_rows": len(causal_rows),
        "conditions": list(CONDITIONS),
        "file_hashes": hashes,
        "interventions": list(INTERVENTIONS),
        "roles": list(ROLES),
        "run_rows": len(run_rows),
        "seed_count": args.seed_count,
        "seed_start": args.seed_start,
    }
    manifest_path = output / "phase4_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksum_paths = sorted((*data_paths, manifest_path), key=lambda path: path.name)
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
