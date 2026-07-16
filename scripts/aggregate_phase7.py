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
    "kv_fixed_position",
    "kv_first_free",
    "kv_hard_router",
    "kv_annealed_router",
)
INTERVENTIONS = (
    "intact", "no_broadcast", "no_workspace_persistence",
    "no_workspace_output", "no_spine_workspace", "no_mechanism_output",
    "workspace_disconnected", "no_workspace_writes", "zero_reader_inbox",
    "permuted_workspace_keys", "zero_query_key", "randomized_write_slots",
    "cleared_writer_assignment", "allow_write_collisions",
    "permuted_recipients",
)
FORMAL_CAUSAL = (
    "randomized_write_slots", "cleared_writer_assignment",
    "allow_write_collisions", "permuted_workspace_keys",
    "zero_query_key", "no_mechanism_output",
)
METRICS = {
    "final_holdout_nll": True,
    "final_holdout_brier": True,
    "final_holdout_ece": True,
    "final_holdout_accuracy": False,
}
CAUSAL_METRICS = {"holdout_nll": True, "holdout_accuracy": False}
PARAMETERS = {
    "structural_core_blind": 725,
    "kv_fixed_position": 88,
    "kv_first_free": 88,
    "kv_hard_router": 112,
    "kv_annealed_router": 112,
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


def load(text: str, width: int = 3) -> list[int]:
    values = [int(value) for value in text.split(";")] if text else []
    if len(values) != width or any(value < 0 for value in values):
        raise ValueError("invalid slot load")
    return values


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    weight = position - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def bootstrap(values: list[float], replicates: int, salt: int) -> tuple[float, float]:
    generator = random.Random(BOOTSTRAP_SEED + salt)
    count = len(values)
    means = [sum(values[generator.randrange(count)] for _ in range(count)) / count
             for _ in range(replicates)]
    return percentile(means, 0.025), percentile(means, 0.975)


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
        raise ValueError("positive seed and bootstrap counts required")
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    expected_raw = {f"{condition}_seed{seed}.csv" for seed in seeds for condition in CONDITIONS}
    expected_causal = {f"causal_seed{seed}.csv" for seed in seeds}
    actual_raw = {path.name for path in args.raw_dir.glob("*.csv")}
    actual_causal = {path.name for path in args.causal_dir.glob("*.csv")}
    if actual_raw != expected_raw:
        raise ValueError(f"raw file set mismatch: {sorted(expected_raw-actual_raw)} {sorted(actual_raw-expected_raw)}")
    if actual_causal != expected_causal:
        raise ValueError("causal file set mismatch")

    run_rows: list[dict[str, str]] = []
    by_key: dict[tuple[int, str], dict[str, str]] = {}
    required = (
        "seed", "steps", "batch_size", "parameters", "condition",
        "training_stream_samples", "read_slot_load", "write_slot_load",
        "final_200_collision_rate", "final_200_routing_entropy",
        "mean_writers_per_token", *METRICS.keys(),
    )
    for seed in seeds:
        for condition in CONDITIONS:
            path = args.raw_dir / f"{condition}_seed{seed}.csv"
            row = single(path)
            if any(field not in row for field in required):
                raise ValueError(f"missing run field in {path}")
            if int(row["seed"]) != seed or row["condition"] != condition:
                raise ValueError("run identity mismatch")
            steps, batch = int(row["steps"]), int(row["batch_size"])
            if int(row["training_stream_samples"]) != steps * batch:
                raise ValueError("online sample count mismatch")
            if int(row["parameters"]) != PARAMETERS[condition]:
                raise ValueError("parameter count mismatch")
            for metric in METRICS:
                finite(row, metric)
            read_load, write_load = load(row["read_slot_load"]), load(row["write_slot_load"])
            if condition != "structural_core_blind":
                holdout = int(row["holdout_count"])
                if sum(read_load) != holdout or sum(write_load) != holdout * 6:
                    raise ValueError("top-1 read/write accounting mismatch")
                if abs(float(row["mean_writers_per_token"]) - 0.75) > 1e-12:
                    raise ValueError("writer budget mismatch")
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
            if int(row["seed"]) != seed or row["condition"] != "kv_annealed_router":
                raise ValueError("causal identity mismatch")
            for metric in CAUSAL_METRICS:
                finite(row, metric)
            load(row["read_slot_load"])
            load(row["write_slot_load"])
            causal_rows.append(row)
            causal_by_key[(seed, row["intervention"])] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase7_runs.csv", list(run_rows[0]), run_rows)
    write_csv(args.output_dir / "phase7_causal.csv", list(causal_rows[0]), causal_rows)

    summary: list[dict[str, object]] = []
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
        for field in ("final_200_collision_rate", "final_200_routing_entropy"):
            out[f"mean_{field}"] = statistics.fmean(float(row[field]) for row in selected)
        summary.append(out)
    write_csv(args.output_dir / "phase7_summary.csv", list(summary[0]), summary)

    comparisons = (
        ("first_free_vs_blind", "structural_core_blind", "kv_first_free"),
        ("annealed_vs_blind", "structural_core_blind", "kv_annealed_router"),
        ("annealed_vs_hard", "kv_hard_router", "kv_annealed_router"),
        ("annealed_vs_fixed", "kv_fixed_position", "kv_annealed_router"),
    )
    paired: list[dict[str, object]] = []
    salt = 10000
    for name, baseline, candidate in comparisons:
        for metric, lower in METRICS.items():
            improvements = []
            for seed in seeds:
                base = float(by_key[(seed, baseline)][metric])
                current = float(by_key[(seed, candidate)][metric])
                improvements.append(base - current if lower else current - base)
            lo, hi = bootstrap(improvements, args.bootstrap_replicates, salt); salt += 1
            paired.append({"comparison": name, "baseline": baseline, "candidate": candidate,
                           "metric": metric, "mean_improvement": statistics.fmean(improvements),
                           "ci_low": lo, "ci_high": hi,
                           "favorable_seeds": sum(value > 0 for value in improvements), "n": len(improvements)})
    write_csv(args.output_dir / "phase7_paired.csv", list(paired[0]), paired)

    causal_comparisons: list[dict[str, object]] = []
    salt = 20000
    for intervention in INTERVENTIONS[1:]:
        for metric, lower in CAUSAL_METRICS.items():
            degradation = []
            for seed in seeds:
                intact = float(causal_by_key[(seed, "intact")][metric])
                current = float(causal_by_key[(seed, intervention)][metric])
                degradation.append(current - intact if lower else intact - current)
            lo, hi = bootstrap(degradation, args.bootstrap_replicates, salt); salt += 1
            causal_comparisons.append({"intervention": intervention, "metric": metric,
                "mean_degradation": statistics.fmean(degradation), "ci_low": lo, "ci_high": hi,
                "harmful_seeds": sum(value > 0 for value in degradation), "n": len(degradation),
                "formal_gate": intervention in FORMAL_CAUSAL})
    write_csv(args.output_dir / "phase7_causal_comparisons.csv",
              list(causal_comparisons[0]), causal_comparisons)

    routing: list[dict[str, object]] = []
    for condition in CONDITIONS:
        selected = [by_key[(seed, condition)] for seed in seeds]
        for kind, field in (("read", "read_slot_load"), ("write", "write_slot_load")):
            slot_totals = [0, 0, 0]
            for row in selected:
                values = load(row[field])
                for index, value in enumerate(values): slot_totals[index] += value
            total = sum(slot_totals)
            for slot, value in enumerate(slot_totals):
                routing.append({"condition": condition, "kind": kind, "slot": slot,
                                "total": value, "share": value / total if total else 0.0})
    write_csv(args.output_dir / "phase7_routing.csv", list(routing[0]), routing)

    paired_index = {(row["comparison"], row["metric"]): row for row in paired}
    causal_index = {(row["intervention"], row["metric"]): row for row in causal_comparisons}
    first_free = [by_key[(seed, "kv_first_free")] for seed in seeds]
    annealed = [by_key[(seed, "kv_annealed_router")] for seed in seeds]
    first_free_gate = all(float(row["final_holdout_accuracy"]) >= 0.99 for row in first_free)
    annealed_gate = (
        statistics.fmean(float(row["final_holdout_accuracy"]) for row in annealed) >= 0.95
        and all(float(paired_index[("annealed_vs_blind", metric)]["ci_low"]) > 0
                for metric in ("final_holdout_nll", "final_holdout_brier", "final_holdout_accuracy"))
    )
    collision_gate = all(float(row["final_200_collision_rate"]) < 0.01 for row in annealed)
    causal_gate = all(float(causal_index[(intervention, metric)]["ci_low"]) > 0
                      for intervention in FORMAL_CAUSAL
                      for metric in ("holdout_nll", "holdout_accuracy"))
    manifest = {
        "analysis_seed": BOOTSTRAP_SEED, "bootstrap_replicates": args.bootstrap_replicates,
        "seed_start": args.seed_start, "seed_count": args.seed_count,
        "conditions": list(CONDITIONS), "interventions": list(INTERVENTIONS),
        "run_rows": len(run_rows), "causal_rows": len(causal_rows),
        "first_free_gate_pass": first_free_gate,
        "annealed_structural_gate_pass": annealed_gate,
        "collision_gate_pass": collision_gate,
        "causal_gate_pass": causal_gate,
        "overall_gate_pass": first_free_gate and annealed_gate and collision_gate and causal_gate,
    }
    (args.output_dir / "phase7_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    names = ("phase7_runs.csv", "phase7_causal.csv", "phase7_summary.csv",
             "phase7_paired.csv", "phase7_causal_comparisons.csv",
             "phase7_routing.csv", "phase7_manifest.json")
    (args.output_dir / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
