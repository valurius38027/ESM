#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONDITIONS = (
    "structural_core_full",
    "structural_core_blind",
    "structural_mediation_linear",
    "structural_mediation_bounded",
)
INTERVENTIONS = (
    "intact", "no_broadcast", "no_workspace_persistence",
    "no_workspace_output", "no_spine_workspace", "no_mechanism_output",
    "workspace_disconnected", "no_workspace_writes", "zero_reader_inbox",
    "permuted_recipients",
)


def write_fixture(root: Path) -> tuple[Path, Path]:
    raw = root / "raw"
    causal = root / "causal"
    raw.mkdir(); causal.mkdir()
    for seed in range(2):
        for index, condition in enumerate(CONDITIONS):
            mediation = "mediation" in condition
            row = {
                "preset": condition,
                "seed": seed, "steps": 800, "batch_size": 8,
                "parameters": 15650 if mediation else (1169 if index == 0 else 725),
                "train_count": 160, "holdout_count": 160,
                "sequence_length": 8, "chance_nll": 1.609437912434,
                "initial_train_nll": 1.61, "initial_holdout_nll": 1.61,
                "final_train_nll": 0.8 - 0.04 * index,
                "final_holdout_nll": 1.5 - 0.05 * index + 0.01 * seed,
                "final_train_accuracy": 0.5 + 0.02 * index,
                "final_holdout_accuracy": 0.3 + 0.03 * index,
                "mean_estimated_madds_per_token": 6784 if mediation else 900,
                "mean_active_mechanisms_per_token": 1 if mediation else 0,
                "mean_writers_per_token": 0.75 if mediation else 0,
                "mean_recipients_per_token": 1 if mediation else 0,
                "first_window_loss": 1.6, "last_window_loss": 0.8,
                "mechanism_load": "4;4;4;4" if mediation else "0;0;0;0;0;0",
                "role_mechanism_load": "entity:1;1;1;1|value:1;1;1;1|filler:0;0;0;0|query_marker:1;0;0;0|query_entity:0;0;0;1" if mediation else "entity:0;0;0;0;0;0|value:0;0;0;0;0;0|filler:0;0;0;0;0;0|query_marker:0;0;0;0;0;0|query_entity:0;0;0;0;0;0",
                "condition": condition,
                "workspace_aux_initial_weight": 0, "workspace_aux_anneal_steps": 0,
                "first_primary_window_loss": 1.6, "last_primary_window_loss": 0.8,
                "last_workspace_aux_window_loss": 0,
                "anneal_boundary_primary_window_loss": 0,
                "final_workspace_aux_weight": 0,
                "final_weighted_workspace_aux_window_loss": 0,
                "training_stream_samples": 6400,
                "output_logit_bound": 1 if condition.endswith("bounded") else 0,
                "initial_train_brier": 0.8, "initial_holdout_brier": 0.8,
                "final_train_brier": 0.7 - 0.03 * index,
                "final_holdout_brier": 0.78 - 0.03 * index,
                "initial_train_ece": 0, "initial_holdout_ece": 0,
                "final_train_ece": 0.2 - 0.02 * index,
                "final_holdout_ece": 0.18 - 0.02 * index,
                "final_train_max_confidence": 0.5,
                "final_holdout_max_confidence": 0.45,
                "final_train_true_class_probability": 0.4,
                "final_holdout_true_class_probability": 0.35 + 0.02 * index,
            }
            path = raw / f"{condition}_seed{seed}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
                writer.writeheader(); writer.writerow(row)

        path = causal / f"causal_seed{seed}.csv"
        rows = []
        for index, intervention in enumerate(INTERVENTIONS):
            rows.append({
                "seed": seed, "intervention": intervention,
                "holdout_nll": 1.3 + 0.03 * index,
                "holdout_accuracy": 0.5 - 0.02 * index,
                "nll_delta_vs_intact": 0.03 * index,
                "accuracy_delta_vs_intact": -0.02 * index,
                "mean_active_mechanisms_per_token": 1,
                "mean_writers_per_token": 0.75,
                "mean_recipients_per_token": 1,
                "mechanism_load": "4;4;4;4",
                "role_mechanism_load": "entity:1;1;1;1|value:1;1;1;1|filler:0;0;0;0|query_marker:1;0;0;0|query_entity:0;0;0;1",
                "condition": "structural_mediation_bounded",
                "holdout_brier": 0.7 + 0.02 * index,
                "holdout_ece": 0.1 + 0.01 * index,
                "holdout_max_confidence": 0.5 - 0.01 * index,
                "holdout_true_class_probability": 0.4 - 0.01 * index,
                "brier_delta_vs_intact": 0.02 * index,
                "ece_delta_vs_intact": 0.01 * index,
                "max_confidence_delta_vs_intact": -0.01 * index,
                "true_class_probability_delta_vs_intact": -0.01 * index,
            })
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    return raw, causal


def run(script: Path, raw: Path, causal: Path, out: Path) -> None:
    subprocess.run([
        sys.executable, str(script), "--raw-dir", str(raw),
        "--causal-dir", str(causal), "--output-dir", str(out),
        "--seed-start", "0", "--seed-count", "2",
        "--bootstrap-replicates", "1000",
    ], check=True)


def bytes_by_name(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(path.iterdir()) if item.is_file()}


def main() -> None:
    source = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="sgw-phase5-") as temp:
        root = Path(temp)
        raw, causal = write_fixture(root)
        first, second = root / "first", root / "second"
        run(source / "scripts/aggregate_phase5.py", raw, causal, first)
        run(source / "scripts/aggregate_phase5.py", raw, causal, second)
        assert bytes_by_name(first) == bytes_by_name(second)
        assert len(list(csv.DictReader((first / "phase5_runs.csv").open()))) == 8
        assert len(list(csv.DictReader((first / "phase5_causal.csv").open()))) == 20
        paired = (first / "phase5_paired.csv").read_text()
        assert "bounded_vs_linear" in paired and "bounded_vs_blind" in paired
        causal_text = (first / "phase5_causal_comparisons.csv").read_text()
        assert "no_workspace_writes" in causal_text and "zero_reader_inbox" in causal_text
        assert b"\r" not in b"".join(bytes_by_name(first).values())
        shutil.copy(raw / "structural_core_full_seed0.csv",
                    raw / "structural_core_full_seed99.csv")
        failed = subprocess.run([
            sys.executable, str(source / "scripts/aggregate_phase5.py"),
            "--raw-dir", str(raw), "--causal-dir", str(causal),
            "--output-dir", str(root / "bad"), "--seed-start", "0",
            "--seed-count", "2", "--bootstrap-replicates", "10",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert failed.returncode != 0


if __name__ == "__main__":
    main()
