#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

CONDITIONS = (
    "structural_core_full",
    "structural_core_blind",
    "structural_mediation_linear",
    "structural_mediation_bounded",
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
LOWER_IS_BETTER = {
    "final_holdout_nll": True,
    "final_holdout_brier": True,
    "final_holdout_ece": True,
    "final_holdout_accuracy": False,
    "final_holdout_max_confidence": False,
    "final_holdout_true_class_probability": False,
}
BOOTSTRAP_SEED = 20260716


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one data row")
    return rows[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_ci(values: list[float], replicates: int, salt: int) -> tuple[float, float]:
    if not values:
        raise ValueError("empty bootstrap input")
    generator = random.Random(BOOTSTRAP_SEED + salt)
    count = len(values)
    means = []
    for _ in range(replicates):
        means.append(sum(values[generator.randrange(count)] for _ in range(count)) / count)
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def parse_load(text: str) -> list[int]:
    if not text:
        return []
    return [int(value) for value in text.split(";")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=50000)
    args = parser.parse_args()
    if args.seed_count <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("seed count and bootstrap replicates must be positive")

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    expected_raw = {
        f"{condition}_seed{seed}.csv" for seed in seeds for condition in CONDITIONS
    }
    actual_raw = {path.name for path in args.raw_dir.glob("*.csv")}
    if actual_raw != expected_raw:
        raise ValueError(f"raw file set mismatch: missing={sorted(expected_raw-actual_raw)} extra={sorted(actual_raw-expected_raw)}")
    expected_causal = {f"causal_seed{seed}.csv" for seed in seeds}
    actual_causal = {path.name for path in args.causal_dir.glob("*.csv")}
    if actual_causal != expected_causal:
        raise ValueError(f"causal file set mismatch: missing={sorted(expected_causal-actual_causal)} extra={sorted(actual_causal-expected_causal)}")

    runs: list[dict[str, str]] = []
    by_key: dict[tuple[int, str], dict[str, str]] = {}
    for seed in seeds:
        for condition in CONDITIONS:
            row = read_single_row(args.raw_dir / f"{condition}_seed{seed}.csv")
            if int(row["seed"]) != seed or row["condition"] != condition:
                raise ValueError("run identity mismatch")
            steps = int(row["steps"])
            batch = int(row["batch_size"])
            if int(row["training_stream_samples"]) != steps * batch:
                raise ValueError("stream sample count does not equal steps times batch")
            expected_bound = 1.0 if condition == "structural_mediation_bounded" else 0.0
            if float(row["output_logit_bound"]) != expected_bound:
                raise ValueError("condition output bound mismatch")
            for field in (
                "final_holdout_nll", "final_holdout_accuracy",
                "final_holdout_brier", "final_holdout_ece",
                "final_holdout_max_confidence",
                "final_holdout_true_class_probability",
            ):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"non-finite {field}")
            runs.append(row)
            by_key[(seed, condition)] = row

    causal: list[dict[str, str]] = []
    causal_by_key: dict[tuple[int, str], dict[str, str]] = {}
    for seed in seeds:
        rows = read_rows(args.causal_dir / f"causal_seed{seed}.csv")
        if len(rows) != len(INTERVENTIONS):
            raise ValueError("causal row count mismatch")
        if {row["intervention"] for row in rows} != set(INTERVENTIONS):
            raise ValueError("causal intervention set mismatch")
        for row in rows:
            if int(row["seed"]) != seed or row["condition"] != "structural_mediation_bounded":
                raise ValueError("causal identity mismatch")
            causal.append(row)
            causal_by_key[(seed, row["intervention"])] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_fields = list(runs[0].keys())
    causal_fields = list(causal[0].keys())
    write_csv(args.output_dir / "phase5_runs.csv", run_fields, runs)
    write_csv(args.output_dir / "phase5_causal.csv", causal_fields, causal)

    metric_fields = list(LOWER_IS_BETTER)
    summary_rows: list[dict[str, object]] = []
    for cindex, condition in enumerate(CONDITIONS):
        selected = [by_key[(seed, condition)] for seed in seeds]
        row: dict[str, object] = {"condition": condition, "n": len(selected)}
        for mindex, metric in enumerate(metric_fields):
            values = [float(item[metric]) for item in selected]
            ci_low, ci_high = bootstrap_ci(
                values, args.bootstrap_replicates, 1000 * cindex + mindex)
            row[f"mean_{metric}"] = sum(values) / len(values)
            row[f"sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"ci_low_{metric}"] = ci_low
            row[f"ci_high_{metric}"] = ci_high
        summary_rows.append(row)
    summary_fields = list(summary_rows[0].keys())
    write_csv(args.output_dir / "phase5_summary.csv", summary_fields, summary_rows)

    comparisons = (
        ("bounded_vs_linear", "structural_mediation_linear", "structural_mediation_bounded"),
        ("bounded_vs_blind", "structural_core_blind", "structural_mediation_bounded"),
        ("bounded_vs_full", "structural_core_full", "structural_mediation_bounded"),
        ("linear_vs_blind", "structural_core_blind", "structural_mediation_linear"),
    )
    paired_rows: list[dict[str, object]] = []
    salt = 10000
    for name, baseline, candidate in comparisons:
        for metric, lower in LOWER_IS_BETTER.items():
            improvements = []
            for seed in seeds:
                baseline_value = float(by_key[(seed, baseline)][metric])
                candidate_value = float(by_key[(seed, candidate)][metric])
                improvements.append(
                    baseline_value - candidate_value if lower
                    else candidate_value - baseline_value
                )
            ci_low, ci_high = bootstrap_ci(
                improvements, args.bootstrap_replicates, salt)
            salt += 1
            paired_rows.append({
                "comparison": name,
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "mean_improvement": sum(improvements) / len(improvements),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "favorable_seeds": sum(value > 0.0 for value in improvements),
                "n": len(improvements),
            })
    paired_fields = list(paired_rows[0].keys())
    write_csv(args.output_dir / "phase5_paired.csv", paired_fields, paired_rows)

    causal_metrics = (
        ("holdout_nll", True),
        ("holdout_accuracy", False),
        ("holdout_brier", True),
        ("holdout_ece", True),
        ("holdout_max_confidence", None),
        ("holdout_true_class_probability", False),
    )
    causal_rows: list[dict[str, object]] = []
    salt = 20000
    for intervention in INTERVENTIONS[1:]:
        for metric, lower in causal_metrics:
            degradations = []
            for seed in seeds:
                intact = float(causal_by_key[(seed, "intact")][metric])
                current = float(causal_by_key[(seed, intervention)][metric])
                if lower is True:
                    degradation = current - intact
                elif lower is False:
                    degradation = intact - current
                else:
                    degradation = current - intact
                degradations.append(degradation)
            ci_low, ci_high = bootstrap_ci(
                degradations, args.bootstrap_replicates, salt)
            salt += 1
            causal_rows.append({
                "intervention": intervention,
                "metric": metric,
                "mean_degradation": sum(degradations) / len(degradations),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "harmful_seeds": sum(value > 0.0 for value in degradations),
                "n": len(degradations),
            })
    causal_comparison_fields = list(causal_rows[0].keys())
    write_csv(args.output_dir / "phase5_causal_comparisons.csv",
              causal_comparison_fields, causal_rows)

    calibration_rows = []
    for condition in CONDITIONS:
        selected = [by_key[(seed, condition)] for seed in seeds]
        mean_accuracy = sum(float(row["final_holdout_accuracy"]) for row in selected) / len(selected)
        mean_confidence = sum(float(row["final_holdout_max_confidence"]) for row in selected) / len(selected)
        calibration_rows.append({
            "condition": condition,
            "mean_accuracy": mean_accuracy,
            "mean_max_confidence": mean_confidence,
            "confidence_minus_accuracy": mean_confidence - mean_accuracy,
            "mean_brier": sum(float(row["final_holdout_brier"]) for row in selected) / len(selected),
            "mean_ece": sum(float(row["final_holdout_ece"]) for row in selected) / len(selected),
            "mean_true_class_probability": sum(float(row["final_holdout_true_class_probability"]) for row in selected) / len(selected),
        })
    write_csv(args.output_dir / "phase5_calibration.csv",
              list(calibration_rows[0].keys()), calibration_rows)

    routing_rows = []
    for condition in CONDITIONS:
        selected = [by_key[(seed, condition)] for seed in seeds]
        loads = [parse_load(row["mechanism_load"]) for row in selected]
        width = len(loads[0])
        if any(len(load) != width for load in loads):
            raise ValueError("mechanism load width mismatch")
        totals = [sum(load[index] for load in loads) for index in range(width)]
        grand_total = sum(totals)
        for mechanism, total in enumerate(totals):
            routing_rows.append({
                "condition": condition,
                "mechanism": mechanism,
                "total_activations": total,
                "activation_share": (total / grand_total) if grand_total else 0.0,
            })
    write_csv(args.output_dir / "phase5_routing.csv",
              list(routing_rows[0].keys()), routing_rows)

    manifest = {
        "analysis_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "conditions": list(CONDITIONS),
        "interventions": list(INTERVENTIONS),
        "run_rows": len(runs),
        "causal_rows": len(causal),
    }
    (args.output_dir / "phase5_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_names = (
        "phase5_runs.csv", "phase5_causal.csv", "phase5_summary.csv",
        "phase5_paired.csv", "phase5_causal_comparisons.csv",
        "phase5_calibration.csv", "phase5_routing.csv", "phase5_manifest.json",
    )
    checksum_lines = []
    for name in output_names:
        digest = hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {name}\n")
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(checksum_lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
