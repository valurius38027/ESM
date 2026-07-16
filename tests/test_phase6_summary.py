#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
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
FORMAL_CAUSAL = {
    "no_workspace_writes",
    "no_workspace_persistence",
    "permuted_workspace_keys",
    "zero_query_key",
    "no_mechanism_output",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_fixture(root: Path) -> tuple[Path, Path]:
    raw = root / "raw"
    causal = root / "causal"
    raw.mkdir()
    causal.mkdir()
    metrics = {
        "structural_core_blind": (1.60, 0.20, 0.80, 0.05),
        "structural_mediation_bounded": (2.50, 0.10, 1.05, 0.30),
        "structural_kv_exact": (0.01, 1.00, 0.0002, 0.004),
        "structural_kv_learned": (0.03, 0.99, 0.002, 0.008),
    }
    parameters = {
        "structural_core_blind": 725,
        "structural_mediation_bounded": 15650,
        "structural_kv_exact": 0,
        "structural_kv_learned": 88,
    }
    for seed in range(30):
        jitter = seed * 0.00001
        for condition in CONDITIONS:
            nll, accuracy, brier, ece = metrics[condition]
            row = {
                "preset": condition,
                "seed": seed,
                "steps": 800,
                "batch_size": 8,
                "parameters": parameters[condition],
                "train_count": 160,
                "holdout_count": 160,
                "sequence_length": 8,
                "chance_nll": 1.609437912434,
                "final_holdout_nll": nll + jitter,
                "final_holdout_accuracy": accuracy,
                "final_holdout_brier": brier + jitter,
                "final_holdout_ece": ece + jitter,
                "final_holdout_max_confidence": accuracy,
                "final_holdout_true_class_probability": accuracy,
                "training_stream_samples": 6400,
                "output_logit_bound": 1 if condition == "structural_mediation_bounded" else 0,
                "mechanism_load": "0" if "core" in condition else "480",
                "read_slot_load": "0;0;0" if "core" in condition else "54;53;53",
                "condition": condition,
            }
            write_csv(raw / f"{condition}_seed{seed}.csv", [row])

        rows: list[dict[str, object]] = []
        for intervention in INTERVENTIONS:
            harmful = intervention in FORMAL_CAUSAL
            rows.append({
                "seed": seed,
                "intervention": intervention,
                "holdout_nll": 1.60 + jitter if harmful else 0.03 + jitter,
                "holdout_accuracy": 0.20 if harmful else 0.99,
                "holdout_brier": 0.80 + jitter if harmful else 0.002 + jitter,
                "holdout_ece": 0.05 + jitter if harmful else 0.008 + jitter,
                "holdout_max_confidence": 0.20 if harmful else 0.99,
                "holdout_true_class_probability": 0.20 if harmful else 0.99,
                "condition": "structural_kv_learned",
                "read_slot_load": "160;0;0" if harmful else "54;53;53",
            })
        write_csv(causal / f"causal_seed{seed}.csv", rows)
    return raw, causal


def run(script: Path, raw: Path, causal: Path, out: Path, *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([
        sys.executable,
        str(script),
        "--raw-dir", str(raw),
        "--causal-dir", str(causal),
        "--output-dir", str(out),
        "--seed-start", "0",
        "--seed-count", "30",
        "--bootstrap-replicates", "1000",
    ], check=check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def bytes_by_name(path: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def main() -> None:
    source = Path(sys.argv[1])
    script = source / "scripts/aggregate_phase6.py"
    with tempfile.TemporaryDirectory(prefix="sgw-phase6-") as temp:
        root = Path(temp)
        raw, causal = write_fixture(root)
        first, second = root / "first", root / "second"
        run(script, raw, causal, first)
        run(script, raw, causal, second)
        assert bytes_by_name(first) == bytes_by_name(second)
        assert len(list(csv.DictReader((first / "phase6_runs.csv").open()))) == 120
        assert len(list(csv.DictReader((first / "phase6_causal.csv").open()))) == 360
        paired = (first / "phase6_paired.csv").read_text()
        assert "exact_vs_blind" in paired
        assert "learned_vs_blind" in paired
        assert "learned_vs_bounded" in paired
        causal_text = (first / "phase6_causal_comparisons.csv").read_text()
        for intervention in FORMAL_CAUSAL:
            assert intervention in causal_text
        routing = (first / "phase6_routing.csv").read_text()
        assert "read_slot" in routing and "structural_kv_learned" in routing
        manifest = json.loads((first / "phase6_manifest.json").read_text())
        assert manifest["capacity_gate_pass"] is True
        assert manifest["learned_structural_gate_pass"] is True
        assert manifest["bounded_control_gate_pass"] is True
        assert manifest["causal_gate_pass"] is True
        assert manifest["overall_gate_pass"] is True
        assert b"\r" not in b"".join(bytes_by_name(first).values())

        shutil.copy(raw / "structural_core_blind_seed0.csv", raw / "extra.csv")
        assert run(script, raw, causal, root / "extra-out", check=False).returncode != 0
        (raw / "extra.csv").unlink()

        missing = raw / "structural_kv_exact_seed0.csv"
        saved = missing.read_bytes()
        missing.unlink()
        assert run(script, raw, causal, root / "missing-out", check=False).returncode != 0
        missing.write_bytes(saved)

        bad = raw / "structural_kv_learned_seed0.csv"
        rows = list(csv.DictReader(bad.open()))
        rows[0]["training_stream_samples"] = "6399"
        write_csv(bad, rows)
        assert run(script, raw, causal, root / "samples-out", check=False).returncode != 0


if __name__ == "__main__":
    main()
