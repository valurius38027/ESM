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
    "kv_full_capacity",
    "kv_oracle_retention",
    "kv_fifo_eviction",
    "kv_reservoir",
    "kv_hard_retention",
    "kv_annealed_retention",
)
INTERVENTIONS = (
    "intact",
    "zero_query_context",
    "randomized_retention_actions",
    "force_fifo_retention",
    "force_relevant_eviction",
    "permuted_context_labels",
    "disable_retention_skip",
    "no_workspace_writes",
    "no_workspace_persistence",
    "no_mechanism_output",
)
FORMAL_CAUSAL = tuple(value for value in INTERVENTIONS if value not in {"intact", "force_fifo_retention"})
METRICS = {
    "final_holdout_nll": True,
    "final_holdout_brier": True,
    "final_holdout_ece": True,
    "final_holdout_accuracy": False,
}
PRIMARY_METRICS = ("final_holdout_nll", "final_holdout_brier", "final_holdout_accuracy")
CAUSAL_METRICS = {"holdout_nll": True, "holdout_accuracy": False}
PARAMETERS = {
    "kv_full_capacity": 120,
    "kv_oracle_retention": 120,
    "kv_fifo_eviction": 120,
    "kv_reservoir": 120,
    "kv_hard_retention": 145,
    "kv_annealed_retention": 145,
}
SLOTS = {
    "kv_full_capacity": 6,
    "kv_oracle_retention": 3,
    "kv_fifo_eviction": 3,
    "kv_reservoir": 3,
    "kv_hard_retention": 3,
    "kv_annealed_retention": 3,
}
BOOTSTRAP_SEED = 20260716


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def single(path: Path) -> dict[str, str]:
    result = rows(path)
    if len(result) != 1:
        raise ValueError(f"{path} must contain one row")
    return result[0]


def write_csv(path: Path, fields: list[str], data: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def load(text: str, width: int) -> list[int]:
    values = [int(value) for value in text.split(";")] if text else []
    if len(values) != width or any(value < 0 for value in values):
        raise ValueError(f"invalid slot load width {width}: {text}")
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def bootstrap(values: list[float], replicates: int, salt: int) -> tuple[float, float]:
    generator = random.Random(BOOTSTRAP_SEED + salt)
    count = len(values)
    means = [sum(values[generator.randrange(count)] for _ in range(count)) / count
             for _ in range(replicates)]
    return percentile(means, 0.025), percentile(means, 0.975)


def improvement(base: float, candidate: float, lower_is_better: bool) -> float:
    return base - candidate if lower_is_better else candidate - base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=50000)
    parser.add_argument("--allow-short-run", action="store_true")
    args = parser.parse_args()
    if args.seed_count <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("positive seed and bootstrap counts required")
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    expected_raw = {f"{condition}_seed{seed}.csv" for seed in seeds for condition in CONDITIONS}
    expected_causal = {f"causal_seed{seed}.csv" for seed in seeds}
    actual_raw = {path.name for path in args.raw_dir.glob("*.csv")}
    actual_causal = {path.name for path in args.causal_dir.glob("*.csv")}
    if actual_raw != expected_raw:
        raise ValueError(f"raw file set mismatch: missing={sorted(expected_raw-actual_raw)} extra={sorted(actual_raw-expected_raw)}")
    if actual_causal != expected_causal:
        raise ValueError(f"causal file set mismatch: missing={sorted(expected_causal-actual_causal)} extra={sorted(actual_causal-expected_causal)}")

    required = (
        "seed", "steps", "batch_size", "parameters", "train_count", "holdout_count",
        "sequence_length", "condition", "training_stream_samples", "read_slot_load",
        "write_slot_load", "eviction_slot_load", "router_anneal_steps",
        "final_200_disagreement_rate", "retention_write_rate", "retention_skip_rate",
        "retention_eviction_rate", "relevant_eviction_rate",
        "queried_entity_retention_rate", "query_read_hit_rate", "mean_retained_age",
        "final_300_retention_write_rate", "final_300_retention_skip_rate",
        "final_300_retention_eviction_rate", "final_300_relevant_eviction_rate",
        "final_300_query_read_hit_rate", *METRICS.keys(),
    )
    run_rows: list[dict[str, str]] = []
    by_key: dict[tuple[int, str], dict[str, str]] = {}
    for seed in seeds:
        for condition in CONDITIONS:
            path = args.raw_dir / f"{condition}_seed{seed}.csv"
            row = single(path)
            if any(field not in row for field in required):
                missing = [field for field in required if field not in row]
                raise ValueError(f"missing run fields in {path}: {missing}")
            if int(row["seed"]) != seed or row["condition"] != condition:
                raise ValueError("run identity mismatch")
            steps, batch = int(row["steps"]), int(row["batch_size"])
            if int(row["training_stream_samples"]) != steps * batch:
                raise ValueError("online sample count mismatch")
            if int(row["parameters"]) != PARAMETERS[condition]:
                raise ValueError("parameter count mismatch")
            if int(row["sequence_length"]) != 15:
                raise ValueError("retention sequence length mismatch")
            for field in METRICS:
                finite(row, field)
            for field in required:
                if field.endswith("_rate") or field == "mean_retained_age":
                    finite(row, field)
            width = SLOTS[condition]
            read_load = load(row["read_slot_load"], width)
            write_load = load(row["write_slot_load"], width)
            eviction_load = load(row["eviction_slot_load"], width)
            holdout = int(row["holdout_count"])
            if sum(read_load) != holdout:
                raise ValueError("top-1 read accounting mismatch")
            if sum(write_load) > holdout * 12 or sum(eviction_load) > holdout * 3:
                raise ValueError("retention accounting exceeds task budget")
            if condition == "kv_annealed_retention" and not args.allow_short_run:
                if int(row["router_anneal_steps"]) != 900 or steps < 1200:
                    raise ValueError("annealed hard-only window mismatch")
                if abs(float(row["final_200_disagreement_rate"])) > 1e-12:
                    raise ValueError("hard-only route disagreement must be zero")
            run_rows.append(row)
            by_key[(seed, condition)] = row

    causal_rows: list[dict[str, str]] = []
    causal_by_key: dict[tuple[int, str], dict[str, str]] = {}
    for seed in seeds:
        path = args.causal_dir / f"causal_seed{seed}.csv"
        current = rows(path)
        if len(current) != len(INTERVENTIONS) or {row["intervention"] for row in current} != set(INTERVENTIONS):
            raise ValueError("causal intervention set mismatch")
        for row in current:
            if int(row["seed"]) != seed or row["condition"] != "kv_annealed_retention":
                raise ValueError("causal identity mismatch")
            for metric in CAUSAL_METRICS:
                finite(row, metric)
            for field in ("retention_write_rate", "retention_skip_rate", "retention_eviction_rate",
                          "relevant_eviction_rate", "queried_entity_retention_rate",
                          "query_read_hit_rate", "mean_retained_age"):
                finite(row, field)
            load(row["read_slot_load"], 3)
            load(row["write_slot_load"], 3)
            load(row["eviction_slot_load"], 3)
            causal_rows.append(row)
            causal_by_key[(seed, row["intervention"])] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase8_runs.csv", list(run_rows[0]), run_rows)
    write_csv(args.output_dir / "phase8_causal.csv", list(causal_rows[0]), causal_rows)

    summary: list[dict[str, object]] = []
    retention_fields = (
        "retention_write_rate", "retention_skip_rate", "retention_eviction_rate",
        "relevant_eviction_rate", "queried_entity_retention_rate", "query_read_hit_rate",
        "mean_retained_age", "final_300_retention_write_rate",
        "final_300_retention_skip_rate", "final_300_retention_eviction_rate",
        "final_300_relevant_eviction_rate", "final_300_query_read_hit_rate",
    )
    for cindex, condition in enumerate(CONDITIONS):
        selected = [by_key[(seed, condition)] for seed in seeds]
        out: dict[str, object] = {"condition": condition, "n": len(selected)}
        for mindex, metric in enumerate(METRICS):
            values = [float(row[metric]) for row in selected]
            lo, hi = bootstrap(values, args.bootstrap_replicates, cindex * 100 + mindex)
            out[f"mean_{metric}"] = statistics.fmean(values)
            out[f"sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
            out[f"ci_low_{metric}"] = lo
            out[f"ci_high_{metric}"] = hi
        for field in retention_fields:
            out[f"mean_{field}"] = statistics.fmean(float(row[field]) for row in selected)
        summary.append(out)
    write_csv(args.output_dir / "phase8_summary.csv", list(summary[0]), summary)

    comparisons = (
        ("annealed_vs_fifo", "kv_fifo_eviction", "kv_annealed_retention"),
        ("annealed_vs_reservoir", "kv_reservoir", "kv_annealed_retention"),
        ("annealed_vs_hard", "kv_hard_retention", "kv_annealed_retention"),
        ("oracle_vs_fifo", "kv_fifo_eviction", "kv_oracle_retention"),
    )
    paired: list[dict[str, object]] = []
    salt = 10000
    for name, baseline, candidate in comparisons:
        for metric, lower in METRICS.items():
            values = [improvement(float(by_key[(seed, baseline)][metric]),
                                  float(by_key[(seed, candidate)][metric]), lower)
                      for seed in seeds]
            lo, hi = bootstrap(values, args.bootstrap_replicates, salt)
            salt += 1
            paired.append({
                "comparison": name, "baseline": baseline, "candidate": candidate,
                "metric": metric, "n": len(values),
                "mean_improvement": statistics.fmean(values),
                "sd_improvement": statistics.stdev(values) if len(values) > 1 else 0.0,
                "ci_low": lo, "ci_high": hi,
                "favorable_count": sum(value > 0 for value in values),
            })
    write_csv(args.output_dir / "phase8_paired.csv", list(paired[0]), paired)

    causal: list[dict[str, object]] = []
    salt = 20000
    for intervention in INTERVENTIONS[1:]:
        for metric, lower in CAUSAL_METRICS.items():
            values = []
            for seed in seeds:
                intact = float(causal_by_key[(seed, "intact")][metric])
                changed = float(causal_by_key[(seed, intervention)][metric])
                values.append(changed - intact if lower else intact - changed)
            lo, hi = bootstrap(values, args.bootstrap_replicates, salt)
            salt += 1
            causal.append({
                "intervention": intervention, "metric": metric, "n": len(values),
                "mean_harm": statistics.fmean(values),
                "sd_harm": statistics.stdev(values) if len(values) > 1 else 0.0,
                "ci_low": lo, "ci_high": hi,
                "harmful_count": sum(value > 0 for value in values),
            })
    write_csv(args.output_dir / "phase8_causal_comparisons.csv", list(causal[0]), causal)

    retention_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        selected = [by_key[(seed, condition)] for seed in seeds]
        retention_rows.append({
            "condition": condition,
            **{f"mean_{field}": statistics.fmean(float(row[field]) for row in selected)
               for field in retention_fields},
        })
    write_csv(args.output_dir / "phase8_retention.csv", list(retention_rows[0]), retention_rows)

    paired_index = {(row["comparison"], row["metric"]): row for row in paired}
    causal_index = {(row["intervention"], row["metric"]): row for row in causal}
    oracle = [by_key[(seed, "kv_oracle_retention")] for seed in seeds]
    annealed = [by_key[(seed, "kv_annealed_retention")] for seed in seeds]
    oracle_gate = all(float(row["final_holdout_accuracy"]) >= 0.99 for row in oracle)
    annealed_accuracy_gate = statistics.fmean(float(row["final_holdout_accuracy"]) for row in annealed) >= 0.90
    fifo_gate = all(float(paired_index[("annealed_vs_fifo", metric)]["ci_low"]) > 0
                    for metric in PRIMARY_METRICS)
    reservoir_gate = all(float(paired_index[("annealed_vs_reservoir", metric)]["ci_low"]) > 0
                         for metric in PRIMARY_METRICS)
    retention_gate = (
        statistics.fmean(float(row["query_read_hit_rate"]) for row in annealed) >= 0.90 and
        statistics.fmean(float(row["relevant_eviction_rate"]) for row in annealed) < 0.05
    )
    formal_run = all(int(row["steps"]) >= 1200 for row in annealed)
    hard_window_gate = (args.allow_short_run and not formal_run) or all(
        int(row["router_anneal_steps"]) == 900 and
        abs(float(row["final_200_disagreement_rate"])) <= 1e-12
        for row in annealed
    )
    causal_gate = all(float(causal_index[(intervention, metric)]["ci_low"]) > 0
                      for intervention in FORMAL_CAUSAL
                      for metric in CAUSAL_METRICS)
    manifest = {
        "analysis_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "conditions": list(CONDITIONS),
        "interventions": list(INTERVENTIONS),
        "formal_causal_interventions": list(FORMAL_CAUSAL),
        "run_rows": len(run_rows),
        "formal_run": formal_run,
        "causal_rows": len(causal_rows),
        "oracle_capacity_gate_pass": oracle_gate,
        "annealed_accuracy_gate_pass": annealed_accuracy_gate,
        "annealed_vs_fifo_gate_pass": fifo_gate,
        "annealed_vs_reservoir_gate_pass": reservoir_gate,
        "retention_quality_gate_pass": retention_gate,
        "hard_only_window_gate_pass": hard_window_gate,
        "causal_gate_pass": causal_gate,
        "overall_gate_pass": oracle_gate and annealed_accuracy_gate and fifo_gate and
                             reservoir_gate and retention_gate and hard_window_gate and causal_gate,
    }
    (args.output_dir / "phase8_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    names = (
        "phase8_runs.csv", "phase8_causal.csv", "phase8_summary.csv",
        "phase8_paired.csv", "phase8_causal_comparisons.csv",
        "phase8_retention.csv", "phase8_manifest.json",
    )
    (args.output_dir / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
