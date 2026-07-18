#!/usr/bin/env python3
"""Strict Phase 25 micro-language-model smoke aggregator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys
from typing import Any

MODELS = ("tiny_rnn", "tiny_gru", "tiny_sgw")
CONTEXTS = (16, 64)
EXPECTED_FILES = {
    "primary.csv",
    "interventions.csv",
    "loss_curve.csv",
    "checkpoint.bin",
    "runtime.csv",
    "SHA256SUMS",
}
DETERMINISTIC_FILES = (
    "primary.csv",
    "interventions.csv",
    "loss_curve.csv",
    "checkpoint.bin",
)


class EvidenceError(RuntimeError):
    pass


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise EvidenceError(f"empty CSV: {path}")
    return rows


def finite_float(row: dict[str, str], key: str, path: pathlib.Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise EvidenceError(f"invalid {key} in {path}") from error
    if not math.isfinite(value):
        raise EvidenceError(f"non-finite {key} in {path}")
    return value


def integer(row: dict[str, str], key: str, path: pathlib.Path) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as error:
        raise EvidenceError(f"invalid integer {key} in {path}") from error


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_unit(unit: pathlib.Path, model: str, context: int) -> dict[str, Any]:
    if not unit.is_dir():
        raise EvidenceError(f"missing unit directory: {unit}")
    names = {entry.name for entry in unit.iterdir() if entry.is_file()}
    if names != EXPECTED_FILES:
        raise EvidenceError(
            f"unexpected file set in {unit}: {sorted(names)}"
        )
    sums: dict[str, str] = {}
    for line in (unit / "SHA256SUMS").read_text().splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] in sums:
            raise EvidenceError(f"malformed SHA256SUMS in {unit}")
        sums[parts[1]] = parts[0]
    if set(sums) != set(DETERMINISTIC_FILES):
        raise EvidenceError(f"incomplete SHA256SUMS in {unit}")
    for name, expected in sums.items():
        if sha256(unit / name) != expected:
            raise EvidenceError(f"SHA-256 mismatch: {unit / name}")

    primary_rows = read_rows(unit / "primary.csv")
    if len(primary_rows) != 1:
        raise EvidenceError(f"primary.csv must contain one row: {unit}")
    primary = primary_rows[0]
    if primary.get("model") != model or integer(primary, "context_length", unit) != context:
        raise EvidenceError(f"unit metadata mismatch: {unit}")
    if integer(primary, "schema_version", unit) != 1:
        raise EvidenceError(f"schema mismatch: {unit}")
    if integer(primary, "seed", unit) != 9000 or integer(primary, "steps", unit) != 300:
        raise EvidenceError(f"frozen training metadata mismatch: {unit}")
    if integer(primary, "batch_size", unit) != 2:
        raise EvidenceError(f"batch-size mismatch: {unit}")
    if not primary.get("corpus_id"):
        raise EvidenceError(f"missing corpus identity: {unit}")
    for key in (
        "initial_validation_nll",
        "final_train_nll",
        "final_validation_nll",
        "final_validation_bpb",
        "estimated_madds_per_predicted_byte",
        "dense_counterfactual_madds_per_predicted_byte",
        "writes_per_predicted_byte",
        "delivered_per_predicted_byte",
    ):
        finite_float(primary, key, unit / "primary.csv")

    interventions = read_rows(unit / "interventions.csv")
    expected_modes = {"learned_sparse"}
    if model == "tiny_sgw":
        expected_modes |= {"no_workspace", "message_permuted", "local_only"}
    modes = {row.get("mode", "") for row in interventions}
    if modes != expected_modes or len(interventions) != len(expected_modes):
        raise EvidenceError(f"intervention mode mismatch: {unit}")
    by_mode: dict[str, dict[str, str]] = {}
    for row in interventions:
        mode = row["mode"]
        by_mode[mode] = row
        for key in (
            "mean_nll",
            "bits_per_byte",
            "accuracy",
            "estimated_madds_per_predicted_byte",
            "dense_counterfactual_madds_per_predicted_byte",
            "writes_per_predicted_byte",
            "delivered_per_predicted_byte",
        ):
            finite_float(row, key, unit / "interventions.csv")

    curve = read_rows(unit / "loss_curve.csv")
    if integer(curve[0], "step", unit) != 0 or integer(curve[-1], "step", unit) != 300:
        raise EvidenceError(f"loss curve endpoints mismatch: {unit}")
    for row in curve:
        finite_float(row, "train_nll", unit / "loss_curve.csv")
        finite_float(row, "validation_nll", unit / "loss_curve.csv")
    return {"primary": primary, "modes": by_mode}


def compare_rerun(primary: pathlib.Path, rerun: pathlib.Path) -> None:
    for model in MODELS:
        for context in CONTEXTS:
            relative = pathlib.Path(f"{model}-context{context}")
            verify_unit(rerun / relative, model, context)
            for name in DETERMINISTIC_FILES:
                if (primary / relative / name).read_bytes() != (rerun / relative / name).read_bytes():
                    raise EvidenceError(f"rerun mismatch: {relative / name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--rerun-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        expected_dirs = {f"{model}-context{context}" for model in MODELS for context in CONTEXTS}
        actual_dirs = {entry.name for entry in args.root.iterdir() if entry.is_dir()}
        if actual_dirs != expected_dirs:
            raise EvidenceError(f"primary unit directory set mismatch: {sorted(actual_dirs)}")
        rerun_dirs = {entry.name for entry in args.rerun_root.iterdir() if entry.is_dir()}
        if rerun_dirs != expected_dirs:
            raise EvidenceError(f"rerun unit directory set mismatch: {sorted(rerun_dirs)}")

        units: dict[tuple[str, int], dict[str, Any]] = {}
        for model in MODELS:
            for context in CONTEXTS:
                units[(model, context)] = verify_unit(
                    args.root / f"{model}-context{context}", model, context
                )
        compare_rerun(args.root, args.rerun_root)

        learning_gates: dict[str, bool] = {}
        bpb_gates: dict[str, bool] = {}
        for (model, context), unit in units.items():
            primary = unit["primary"]
            initial = finite_float(primary, "initial_validation_nll", args.root)
            final = finite_float(primary, "final_validation_nll", args.root)
            bpb = finite_float(primary, "final_validation_bpb", args.root)
            name = f"{model}-context{context}"
            learning_gates[name] = final <= initial * 0.98
            bpb_gates[name] = bpb < 8.0

        causal_by_context: dict[str, bool] = {}
        route_by_context: dict[str, bool] = {}
        compute_by_context: dict[str, bool] = {}
        for context in CONTEXTS:
            unit = units[("tiny_sgw", context)]
            modes = unit["modes"]
            learned = finite_float(modes["learned_sparse"], "bits_per_byte", args.root)
            no_workspace = finite_float(modes["no_workspace"], "bits_per_byte", args.root)
            permuted = finite_float(modes["message_permuted"], "bits_per_byte", args.root)
            causal_by_context[str(context)] = (
                no_workspace - learned >= 0.02 and permuted - learned >= 0.02
            )
            primary = unit["primary"]
            writes = finite_float(primary, "writes_per_predicted_byte", args.root)
            delivered = finite_float(primary, "delivered_per_predicted_byte", args.root)
            route_by_context[str(context)] = 0.0 < writes <= 1.0 and 0.0 < delivered <= 1.0
            sparse = finite_float(primary, "estimated_madds_per_predicted_byte", args.root)
            dense = finite_float(primary, "dense_counterfactual_madds_per_predicted_byte", args.root)
            compute_by_context[str(context)] = sparse < dense

        context16 = finite_float(
            units[("tiny_sgw", 16)]["primary"], "final_validation_bpb", args.root
        )
        context64 = finite_float(
            units[("tiny_sgw", 64)]["primary"], "final_validation_bpb", args.root
        )
        context_gate = context64 <= context16 + 0.05
        workspace_causal_gate = any(causal_by_context.values())
        all_gates = (
            all(learning_gates.values())
            and all(bpb_gates.values())
            and workspace_causal_gate
            and context_gate
            and all(route_by_context.values())
            and all(compute_by_context.values())
        )
        gate = {
            "schema_version": 1,
            "seed": 9000,
            "contexts": list(CONTEXTS),
            "models": list(MODELS),
            "deterministic_evidence_pass": True,
            "learning_gates": learning_gates,
            "uniform_bpb_gates": bpb_gates,
            "workspace_causal_by_context": causal_by_context,
            "workspace_causal_gate": workspace_causal_gate,
            "context_scaling_gate": context_gate,
            "route_bounds_by_context": route_by_context,
            "sparse_compute_by_context": compute_by_context,
            "all_gates_pass": all_gates,
            "three_seed_replication_authorized": all_gates,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "lmv1_phase25_gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n"
        )
        with (args.output_dir / "lmv1_phase25_summary.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["model", "context", "initial_validation_nll", "final_validation_nll", "final_validation_bpb"])
            for model in MODELS:
                for context in CONTEXTS:
                    primary = units[(model, context)]["primary"]
                    writer.writerow([model, context, primary["initial_validation_nll"], primary["final_validation_nll"], primary["final_validation_bpb"]])
        lines = [
            "# LMV1 Phase 25 micro smoke",
            "",
            f"- all gates pass: `{str(all_gates).lower()}`",
            f"- workspace causal gate: `{str(workspace_causal_gate).lower()}`",
            f"- context scaling gate: `{str(context_gate).lower()}`",
            "",
        ]
        (args.output_dir / "LMV1_PHASE25_SUMMARY.md").write_text("\n".join(lines))
        return 0
    except (EvidenceError, FileNotFoundError, OSError) as error:
        print(f"aggregate_lmv1_phase25_smoke: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
