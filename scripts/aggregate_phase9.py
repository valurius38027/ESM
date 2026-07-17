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
    "kv_delayed_oracle",
    "kv_delayed_fifo",
    "kv_delayed_reservoir",
    "kv_delayed_hard",
    "kv_delayed_annealed_direct",
    "kv_delayed_annealed_curriculum",
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
    "remove_delay_distractors",
    "relevant_looking_delay_distractors",
    "reverse_delay_block",
)
FORMAL_CAUSAL = (
    "zero_query_context",
    "randomized_retention_actions",
    "force_relevant_eviction",
    "no_workspace_writes",
    "no_workspace_persistence",
    "no_mechanism_output",
    "relevant_looking_delay_distractors",
)
METRICS = {
    "final_holdout_nll": True,
    "final_holdout_brier": True,
    "final_holdout_ece": True,
    "final_holdout_accuracy": False,
}
PRIMARY = ("final_holdout_nll", "final_holdout_brier", "final_holdout_accuracy")
CAUSAL_METRICS = {"holdout_nll": True, "holdout_accuracy": False}
PARAMETERS = {
    "kv_delayed_oracle": 120,
    "kv_delayed_fifo": 120,
    "kv_delayed_reservoir": 120,
    "kv_delayed_hard": 145,
    "kv_delayed_annealed_direct": 145,
    "kv_delayed_annealed_curriculum": 145,
}
DELAYS = (0, 6, 12, 18, 24)
BOOTSTRAP_SEED = 20260717


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_single(path: Path) -> dict[str, str]:
    values = read_rows(path)
    if len(values) != 1:
        raise ValueError(f"{path} must contain exactly one row")
    return values[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], field: str) -> float:
    if field not in row:
        raise ValueError(f"missing field {field}")
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def slot_load(text: str, width: int) -> list[int]:
    values = [int(value) for value in text.split(";")] if text else []
    if len(values) != width or any(value < 0 for value in values):
        raise ValueError(f"invalid slot load: {text}")
    return values


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
        "final_450_disagreement_rate", "relevant_eviction_rate", "query_read_hit_rate",
        "distractor_write_rate", "distractor_eviction_rate", "relevant_survival_rate",
        "final_450_distractor_write_rate", "final_450_distractor_eviction_rate",
        "final_450_relevant_survival_rate", "delay_binding_count",
        "mean_source_query_distance", *METRICS.keys(),
        *(f"delay{delay}_{suffix}" for delay in DELAYS
          for suffix in ("nll", "accuracy", "query_hit", "survival")),
    )
    run_rows: list[dict[str, str]] = []
    by_key: dict[tuple[int, str], dict[str, str]] = {}
    for seed in seeds:
        for condition in CONDITIONS:
            path = args.raw_dir / f"{condition}_seed{seed}.csv"
            row = read_single(path)
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(f"missing run fields in {path}: {missing}")
            if int(row["seed"]) != seed or row["condition"] != condition:
                raise ValueError("run identity mismatch")
            steps = int(row["steps"])
            batch = int(row["batch_size"])
            if int(row["training_stream_samples"]) != steps * batch:
                raise ValueError("online sample count mismatch")
            if int(row["parameters"]) != PARAMETERS[condition]:
                raise ValueError("parameter count mismatch")
            if int(row["sequence_length"]) != 51 or abs(number(row, "delay_binding_count") - 18.0) > 1e-12:
                raise ValueError("formal delayed sequence mismatch")
            if number(row, "mean_source_query_distance") < 38.0:
                raise ValueError("source-query distance too short")
            if sum(slot_load(row["read_slot_load"], 3)) != int(row["holdout_count"]):
                raise ValueError("top-1 read accounting mismatch")
            slot_load(row["write_slot_load"], 3)
            slot_load(row["eviction_slot_load"], 3)
            for field in required:
                if field not in {"condition", "read_slot_load", "write_slot_load", "eviction_slot_load"}:
                    number(row, field)
            if condition in {"kv_delayed_annealed_direct", "kv_delayed_annealed_curriculum"} and not args.allow_short_run:
                if int(row["router_anneal_steps"]) != 1350 or steps < 1800:
                    raise ValueError("phase9 annealing schedule mismatch")
                if abs(number(row, "final_450_disagreement_rate")) > 1e-12:
                    raise ValueError("final hard-only disagreement must be zero")
            run_rows.append(row)
            by_key[(seed, condition)] = row

    causal_rows: list[dict[str, str]] = []
    causal_by_key: dict[tuple[int, str], dict[str, str]] = {}
    for seed in seeds:
        path = args.causal_dir / f"causal_seed{seed}.csv"
        values = read_rows(path)
        if len(values) != len(INTERVENTIONS) or {row["intervention"] for row in values} != set(INTERVENTIONS):
            raise ValueError("causal intervention set mismatch")
        for row in values:
            if int(row["seed"]) != seed or row["condition"] != "kv_delayed_annealed_curriculum":
                raise ValueError("causal identity mismatch")
            for field in (*CAUSAL_METRICS, "distractor_write_rate", "distractor_eviction_rate", "relevant_survival_rate"):
                number(row, field)
            slot_load(row["read_slot_load"], 3)
            slot_load(row["write_slot_load"], 3)
            slot_load(row["eviction_slot_load"], 3)
            causal_rows.append(row)
            causal_by_key[(seed, row["intervention"])] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase9_runs.csv", run_rows)
    write_csv(args.output_dir / "phase9_causal.csv", causal_rows)

    summary_fields = (
        "relevant_eviction_rate", "query_read_hit_rate", "distractor_write_rate",
        "distractor_eviction_rate", "relevant_survival_rate",
        "final_450_distractor_write_rate", "final_450_distractor_eviction_rate",
        "final_450_relevant_survival_rate", "final_450_disagreement_rate",
    )
    summary: list[dict[str, object]] = []
    for cindex, condition in enumerate(CONDITIONS):
        selected = [by_key[(seed, condition)] for seed in seeds]
        output: dict[str, object] = {"condition": condition, "n": len(selected)}
        for mindex, metric in enumerate(METRICS):
            values = [number(row, metric) for row in selected]
            lo, hi = bootstrap(values, args.bootstrap_replicates, cindex * 100 + mindex)
            output[f"mean_{metric}"] = statistics.fmean(values)
            output[f"sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
            output[f"ci_low_{metric}"] = lo
            output[f"ci_high_{metric}"] = hi
        for field in summary_fields:
            output[f"mean_{field}"] = statistics.fmean(number(row, field) for row in selected)
        output["accuracy_ge_90_count"] = sum(number(row, "final_holdout_accuracy") >= 0.90 for row in selected)
        output["minimum_accuracy"] = min(number(row, "final_holdout_accuracy") for row in selected)
        summary.append(output)
    write_csv(args.output_dir / "phase9_summary.csv", summary)

    comparisons = (
        ("curriculum_vs_fifo", "kv_delayed_fifo", "kv_delayed_annealed_curriculum"),
        ("curriculum_vs_reservoir", "kv_delayed_reservoir", "kv_delayed_annealed_curriculum"),
        ("curriculum_vs_hard", "kv_delayed_hard", "kv_delayed_annealed_curriculum"),
        ("curriculum_vs_direct", "kv_delayed_annealed_direct", "kv_delayed_annealed_curriculum"),
        ("direct_vs_hard", "kv_delayed_hard", "kv_delayed_annealed_direct"),
    )
    paired_metrics = {**METRICS, "query_read_hit_rate": False, "relevant_survival_rate": False}
    paired: list[dict[str, object]] = []
    salt = 10000
    for name, baseline, candidate in comparisons:
        for metric, lower in paired_metrics.items():
            values = [improvement(number(by_key[(seed, baseline)], metric),
                                  number(by_key[(seed, candidate)], metric), lower)
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
    write_csv(args.output_dir / "phase9_paired.csv", paired)

    causal: list[dict[str, object]] = []
    salt = 20000
    for intervention in INTERVENTIONS[1:]:
        for metric, lower in CAUSAL_METRICS.items():
            harms = []
            for seed in seeds:
                intact = number(causal_by_key[(seed, "intact")], metric)
                changed = number(causal_by_key[(seed, intervention)], metric)
                harms.append(changed - intact if lower else intact - changed)
            lo, hi = bootstrap(harms, args.bootstrap_replicates, salt)
            salt += 1
            causal.append({
                "intervention": intervention, "metric": metric, "n": len(harms),
                "mean_harm": statistics.fmean(harms),
                "sd_harm": statistics.stdev(harms) if len(harms) > 1 else 0.0,
                "ci_low": lo, "ci_high": hi,
                "harmful_count": sum(value > 0 for value in harms),
            })
    write_csv(args.output_dir / "phase9_causal_comparisons.csv", causal)

    retention = [{
        "condition": condition,
        **{f"mean_{field}": statistics.fmean(number(by_key[(seed, condition)], field) for seed in seeds)
           for field in summary_fields},
    } for condition in CONDITIONS]
    write_csv(args.output_dir / "phase9_retention.csv", retention)

    delay_curves: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for delay in DELAYS:
            selected = [by_key[(seed, condition)] for seed in seeds]
            delay_curves.append({
                "condition": condition,
                "delay": delay,
                "mean_nll": statistics.fmean(number(row, f"delay{delay}_nll") for row in selected),
                "mean_accuracy": statistics.fmean(number(row, f"delay{delay}_accuracy") for row in selected),
                "mean_query_hit": statistics.fmean(number(row, f"delay{delay}_query_hit") for row in selected),
                "mean_survival": statistics.fmean(number(row, f"delay{delay}_survival") for row in selected),
            })
    write_csv(args.output_dir / "phase9_delay_curves.csv", delay_curves)

    paired_index = {(row["comparison"], row["metric"]): row for row in paired}
    causal_index = {(row["intervention"], row["metric"]): row for row in causal}
    oracle = [by_key[(seed, "kv_delayed_oracle")] for seed in seeds]
    curriculum = [by_key[(seed, "kv_delayed_annealed_curriculum")] for seed in seeds]
    formal_run = all(int(row["steps"]) >= 1800 for row in curriculum)
    oracle_gate = all(number(row, "delay18_accuracy") >= 0.99 and
                      number(row, "delay24_accuracy") >= 0.99 for row in oracle)
    capability_gate = (
        statistics.fmean(number(row, "delay18_accuracy") for row in curriculum) >= 0.90 and
        sum(number(row, "delay18_accuracy") >= 0.90 for row in curriculum) >= 27 and
        min(number(row, "delay18_accuracy") for row in curriculum) >= 0.80 and
        statistics.fmean(number(row, "delay24_accuracy") for row in curriculum) >= 0.85 and
        statistics.fmean(number(row, "delay18_query_hit") for row in curriculum) >= 0.90 and
        statistics.fmean(number(row, "delay18_survival") for row in curriculum) >= 0.90 and
        statistics.fmean(number(row, "relevant_eviction_rate") for row in curriculum) < 0.05
    )
    hard_gate = (args.allow_short_run and not formal_run) or all(
        int(row["router_anneal_steps"]) == 1350 and
        abs(number(row, "final_450_disagreement_rate")) <= 1e-12
        for row in curriculum
    )
    baseline_gate = all(
        number(paired_index[(comparison, metric)], "ci_low") > 0
        for comparison in ("curriculum_vs_fifo", "curriculum_vs_reservoir", "curriculum_vs_hard")
        for metric in PRIMARY
    )
    curriculum_positive = sum(
        number(paired_index[("curriculum_vs_direct", metric)], "ci_low") > 0
        for metric in (*PRIMARY, "query_read_hit_rate", "relevant_survival_rate")
    )
    curriculum_not_worse = all(
        number(paired_index[("curriculum_vs_direct", metric)], "ci_high") >= 0
        for metric in PRIMARY
    )
    curriculum_gate = curriculum_positive >= 2 and curriculum_not_worse
    causal_gate = all(
        number(causal_index[(intervention, metric)], "ci_low") > 0
        for intervention in FORMAL_CAUSAL
        for metric in CAUSAL_METRICS
    )
    manifest = {
        "analysis_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "conditions": list(CONDITIONS),
        "interventions": list(INTERVENTIONS),
        "run_rows": len(run_rows),
        "causal_rows": len(causal_rows),
        "formal_run": formal_run,
        "oracle_capacity_gate_pass": oracle_gate,
        "long_horizon_capability_gate_pass": capability_gate,
        "hard_only_window_gate_pass": hard_gate,
        "baseline_comparison_gate_pass": baseline_gate,
        "curriculum_advantage_gate_pass": curriculum_gate,
        "causal_gate_pass": causal_gate,
        "overall_gate_pass": oracle_gate and capability_gate and hard_gate and baseline_gate and curriculum_gate and causal_gate,
    }
    (args.output_dir / "phase9_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    names = (
        "phase9_runs.csv", "phase9_causal.csv", "phase9_summary.csv",
        "phase9_paired.csv", "phase9_causal_comparisons.csv",
        "phase9_retention.csv", "phase9_delay_curves.csv", "phase9_manifest.json",
    )
    (args.output_dir / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
