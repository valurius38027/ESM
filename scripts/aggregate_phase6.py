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
    "structural_core_blind",
    "structural_mediation_bounded",
    "structural_kv_exact",
    "structural_kv_learned",
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
    "permuted_workspace_keys",
    "zero_query_key",
    "permuted_recipients",
)
FORMAL_CAUSAL_INTERVENTIONS = (
    "no_workspace_writes",
    "no_workspace_persistence",
    "permuted_workspace_keys",
    "zero_query_key",
    "no_mechanism_output",
)
LOWER_IS_BETTER = {
    "final_holdout_nll": True,
    "final_holdout_brier": True,
    "final_holdout_ece": True,
    "final_holdout_accuracy": False,
    "final_holdout_max_confidence": False,
    "final_holdout_true_class_probability": False,
}
CAUSAL_METRICS = {
    "holdout_nll": True,
    "holdout_accuracy": False,
    "holdout_brier": True,
    "holdout_ece": True,
    "holdout_max_confidence": False,
    "holdout_true_class_probability": False,
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


def require_fields(row: dict[str, str], fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{context} missing fields: {missing}")


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
    means = [
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    ]
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def parse_load(text: str, *, width: int | None = None) -> list[int]:
    values = [] if not text else [int(value) for value in text.split(";")]
    if width is not None and len(values) != width:
        raise ValueError(f"load width mismatch: expected {width}, got {len(values)}")
    if any(value < 0 for value in values):
        raise ValueError("load values must be nonnegative")
    return values


def finite_float(row: dict[str, str], field: str, context: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in {context}")
    return value


def comparison_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["comparison"]), str(row["metric"])): row for row in rows}


def causal_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["intervention"]), str(row["metric"])): row for row in rows}


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
        raise ValueError(
            f"raw file set mismatch: missing={sorted(expected_raw-actual_raw)} "
            f"extra={sorted(actual_raw-expected_raw)}"
        )
    expected_causal = {f"causal_seed{seed}.csv" for seed in seeds}
    actual_causal = {path.name for path in args.causal_dir.glob("*.csv")}
    if actual_causal != expected_causal:
        raise ValueError(
            f"causal file set mismatch: missing={sorted(expected_causal-actual_causal)} "
            f"extra={sorted(actual_causal-expected_causal)}"
        )

    required_run_fields = (
        "seed", "steps", "batch_size", "parameters", "condition",
        "training_stream_samples", "output_logit_bound", "mechanism_load",
        "read_slot_load", *LOWER_IS_BETTER.keys(),
    )
    runs: list[dict[str, str]] = []
    by_key: dict[tuple[int, str], dict[str, str]] = {}
    for seed in seeds:
        for condition in CONDITIONS:
            path = args.raw_dir / f"{condition}_seed{seed}.csv"
            row = read_single_row(path)
            require_fields(row, required_run_fields, str(path))
            if int(row["seed"]) != seed or row["condition"] != condition:
                raise ValueError("run identity mismatch")
            steps = int(row["steps"])
            batch = int(row["batch_size"])
            if steps <= 0 or batch <= 0:
                raise ValueError("steps and batch size must be positive")
            if int(row["training_stream_samples"]) != steps * batch:
                raise ValueError("stream sample count does not equal steps times batch")
            expected_bound = 1.0 if condition == "structural_mediation_bounded" else 0.0
            if float(row["output_logit_bound"]) != expected_bound:
                raise ValueError("condition output bound mismatch")
            parameter_count = int(row["parameters"])
            if condition == "structural_kv_exact" and parameter_count != 0:
                raise ValueError("exact key-value condition must have zero parameters")
            if condition == "structural_kv_learned" and parameter_count != 88:
                raise ValueError("learned key-value condition must have 88 parameters")
            for field in LOWER_IS_BETTER:
                finite_float(row, field, str(path))
            parse_load(row["mechanism_load"])
            parse_load(row["read_slot_load"], width=3)
            runs.append(row)
            by_key[(seed, condition)] = row

    required_causal_fields = (
        "seed", "intervention", "condition", "read_slot_load",
        *CAUSAL_METRICS.keys(),
    )
    causal: list[dict[str, str]] = []
    causal_by_key: dict[tuple[int, str], dict[str, str]] = {}
    for seed in seeds:
        path = args.causal_dir / f"causal_seed{seed}.csv"
        rows = read_rows(path)
        if len(rows) != len(INTERVENTIONS):
            raise ValueError("causal row count mismatch")
        if {row.get("intervention") for row in rows} != set(INTERVENTIONS):
            raise ValueError("causal intervention set mismatch")
        for row in rows:
            require_fields(row, required_causal_fields, str(path))
            if int(row["seed"]) != seed or row["condition"] != "structural_kv_learned":
                raise ValueError("causal identity mismatch")
            for field in CAUSAL_METRICS:
                finite_float(row, field, str(path))
            parse_load(row["read_slot_load"], width=3)
            causal.append(row)
            causal_by_key[(seed, row["intervention"])] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase6_runs.csv", list(runs[0]), runs)
    write_csv(args.output_dir / "phase6_causal.csv", list(causal[0]), causal)

    summary_rows: list[dict[str, object]] = []
    for cindex, condition in enumerate(CONDITIONS):
        selected = [by_key[(seed, condition)] for seed in seeds]
        row: dict[str, object] = {"condition": condition, "n": len(selected)}
        for mindex, metric in enumerate(LOWER_IS_BETTER):
            values = [float(item[metric]) for item in selected]
            ci_low, ci_high = bootstrap_ci(
                values, args.bootstrap_replicates, 1000 * cindex + mindex
            )
            row[f"mean_{metric}"] = sum(values) / len(values)
            row[f"sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"ci_low_{metric}"] = ci_low
            row[f"ci_high_{metric}"] = ci_high
        summary_rows.append(row)
    write_csv(args.output_dir / "phase6_summary.csv", list(summary_rows[0]), summary_rows)

    comparisons = (
        ("exact_vs_blind", "structural_core_blind", "structural_kv_exact"),
        ("learned_vs_blind", "structural_core_blind", "structural_kv_learned"),
        ("learned_vs_bounded", "structural_mediation_bounded", "structural_kv_learned"),
        ("learned_vs_exact", "structural_kv_exact", "structural_kv_learned"),
    )
    paired_rows: list[dict[str, object]] = []
    salt = 10000
    for name, baseline, candidate in comparisons:
        for metric, lower in LOWER_IS_BETTER.items():
            improvements = []
            for seed in seeds:
                base = float(by_key[(seed, baseline)][metric])
                current = float(by_key[(seed, candidate)][metric])
                improvements.append(base - current if lower else current - base)
            ci_low, ci_high = bootstrap_ci(improvements, args.bootstrap_replicates, salt)
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
    write_csv(args.output_dir / "phase6_paired.csv", list(paired_rows[0]), paired_rows)

    causal_rows: list[dict[str, object]] = []
    salt = 20000
    for intervention in INTERVENTIONS[1:]:
        for metric, lower in CAUSAL_METRICS.items():
            degradations = []
            for seed in seeds:
                intact = float(causal_by_key[(seed, "intact")][metric])
                current = float(causal_by_key[(seed, intervention)][metric])
                degradations.append(current - intact if lower else intact - current)
            ci_low, ci_high = bootstrap_ci(degradations, args.bootstrap_replicates, salt)
            salt += 1
            causal_rows.append({
                "intervention": intervention,
                "metric": metric,
                "mean_degradation": sum(degradations) / len(degradations),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "harmful_seeds": sum(value > 0.0 for value in degradations),
                "n": len(degradations),
                "formal_gate": intervention in FORMAL_CAUSAL_INTERVENTIONS,
            })
    write_csv(
        args.output_dir / "phase6_causal_comparisons.csv",
        list(causal_rows[0]),
        causal_rows,
    )

    routing_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        selected = [by_key[(seed, condition)] for seed in seeds]
        loads = [parse_load(row["read_slot_load"], width=3) for row in selected]
        totals = [sum(load[slot] for load in loads) for slot in range(3)]
        grand_total = sum(totals)
        for slot, total in enumerate(totals):
            routing_rows.append({
                "condition": condition,
                "read_slot": slot,
                "total_reads": total,
                "read_share": total / grand_total if grand_total else 0.0,
            })
    write_csv(args.output_dir / "phase6_routing.csv", list(routing_rows[0]), routing_rows)

    paired_index = comparison_lookup(paired_rows)
    causal_index = causal_lookup(causal_rows)
    exact_rows = [by_key[(seed, "structural_kv_exact")] for seed in seeds]
    learned_rows = [by_key[(seed, "structural_kv_learned")] for seed in seeds]
    capacity_gate = all(
        float(row["final_holdout_accuracy"]) >= 0.99
        and float(row["final_holdout_nll"]) < 0.05
        for row in exact_rows
    )
    favorable_required = 28 if args.seed_count >= 30 else args.seed_count
    learned_metrics = (
        "final_holdout_nll",
        "final_holdout_brier",
        "final_holdout_accuracy",
    )
    learned_structural_gate = (
        sum(float(row["final_holdout_accuracy"]) for row in learned_rows) / len(learned_rows)
        >= 0.95
        and all(
            float(paired_index[("learned_vs_blind", metric)]["ci_low"]) > 0.0
            and int(paired_index[("learned_vs_blind", metric)]["favorable_seeds"])
            >= favorable_required
            for metric in learned_metrics
        )
    )
    bounded_control_gate = all(
        float(paired_index[("learned_vs_bounded", metric)]["ci_low"]) > 0.0
        for metric in ("final_holdout_nll", "final_holdout_accuracy")
    )
    causal_gate = all(
        float(causal_index[(intervention, metric)]["ci_low"]) > 0.0
        for intervention in FORMAL_CAUSAL_INTERVENTIONS
        for metric in ("holdout_nll", "holdout_accuracy")
    )

    manifest = {
        "analysis_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "conditions": list(CONDITIONS),
        "interventions": list(INTERVENTIONS),
        "formal_causal_interventions": list(FORMAL_CAUSAL_INTERVENTIONS),
        "run_rows": len(runs),
        "causal_rows": len(causal),
        "capacity_gate_pass": capacity_gate,
        "learned_structural_gate_pass": learned_structural_gate,
        "bounded_control_gate_pass": bounded_control_gate,
        "causal_gate_pass": causal_gate,
        "overall_gate_pass": (
            capacity_gate and learned_structural_gate and bounded_control_gate and causal_gate
        ),
    }
    (args.output_dir / "phase6_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    output_names = (
        "phase6_runs.csv",
        "phase6_causal.csv",
        "phase6_summary.csv",
        "phase6_paired.csv",
        "phase6_causal_comparisons.csv",
        "phase6_routing.csv",
        "phase6_manifest.json",
    )
    checksum_lines = []
    for name in output_names:
        digest = hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {name}\n")
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(checksum_lines), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
