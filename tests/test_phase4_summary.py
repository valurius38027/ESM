#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
RUN_FIELDS = (
    "preset,seed,steps,batch_size,parameters,train_count,holdout_count,"
    "sequence_length,chance_nll,initial_train_nll,initial_holdout_nll,"
    "final_train_nll,final_holdout_nll,final_train_accuracy,"
    "final_holdout_accuracy,mean_estimated_madds_per_token,"
    "mean_active_mechanisms_per_token,mean_writers_per_token,"
    "mean_recipients_per_token,first_window_loss,last_window_loss,"
    "mechanism_load,role_mechanism_load,condition,"
    "workspace_aux_initial_weight,workspace_aux_anneal_steps,"
    "first_primary_window_loss,last_primary_window_loss,"
    "last_workspace_aux_window_loss,anneal_boundary_primary_window_loss,"
    "final_workspace_aux_weight,final_weighted_workspace_aux_window_loss"
).split(",")
CAUSAL_FIELDS = (
    "seed,intervention,holdout_nll,holdout_accuracy,nll_delta_vs_intact,"
    "accuracy_delta_vs_intact,mean_active_mechanisms_per_token,"
    "mean_writers_per_token,mean_recipients_per_token,mechanism_load,"
    "role_mechanism_load,condition"
).split(",")


def role_text(seed: int, mechanisms: int, active: bool) -> str:
    names = ("entity", "value", "filler", "query_marker", "query_entity")
    rows = []
    for role, name in enumerate(names):
        counts = [0] * mechanisms
        if active:
            for mechanism in range(mechanisms):
                counts[mechanism] = 1 + ((seed + role + mechanism) % 4)
        rows.append(name + ":" + ";".join(str(value) for value in counts))
    return "|".join(rows)


def write_fixture(root: Path) -> tuple[Path, Path]:
    raw = root / "raw"
    causal = root / "causal"
    raw.mkdir()
    causal.mkdir()
    params = (1169, 725, 15650, 15650)
    madds = (984, 696, 6784, 6784)
    for seed in range(2):
        for index, condition in enumerate(CONDITIONS):
            mediation = condition.startswith("mediation")
            aux = condition == "mediation_aux_annealed"
            row = {
                "preset": "mediation_fixed" if mediation else condition,
                "seed": seed,
                "steps": 400,
                "batch_size": 8,
                "parameters": params[index],
                "train_count": 160,
                "holdout_count": 48,
                "sequence_length": 8,
                "chance_nll": 1.609437912434,
                "initial_train_nll": 1.65,
                "initial_holdout_nll": 1.65,
                "final_train_nll": 0.5 - 0.03 * index,
                "final_holdout_nll": 1.6 - 0.08 * index + 0.02 * seed,
                "final_train_accuracy": 0.8,
                "final_holdout_accuracy": 0.4 + 0.03 * index,
                "mean_estimated_madds_per_token": madds[index],
                "mean_active_mechanisms_per_token": 1 if mediation else 0,
                "mean_writers_per_token": 0.75 if mediation else 0,
                "mean_recipients_per_token": 1 if mediation else 0,
                "first_window_loss": 1.6,
                "last_window_loss": 0.5,
                "mechanism_load": "4;4;4;4" if mediation else "0;0;0;0;0;0",
                "role_mechanism_load": role_text(seed, 4, mediation)
                if mediation else role_text(seed, 6, False),
                "condition": condition,
                "workspace_aux_initial_weight": 0.5 if aux else 0.0,
                "workspace_aux_anneal_steps": 200 if aux else 0,
                "first_primary_window_loss": 1.6,
                "last_primary_window_loss": 0.5,
                "last_workspace_aux_window_loss": 0.6 if mediation else 0.0,
                "anneal_boundary_primary_window_loss": 0.7 if aux else 0.0,
                "final_workspace_aux_weight": 0.0,
                "final_weighted_workspace_aux_window_loss": 0.0,
            }
            with (raw / f"{condition}_seed{seed}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerow(row)

        with (causal / f"causal_seed{seed}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=CAUSAL_FIELDS, lineterminator="\n")
            writer.writeheader()
            for index, intervention in enumerate(INTERVENTIONS):
                writer.writerow({
                    "seed": seed,
                    "intervention": intervention,
                    "holdout_nll": 1.3 + 0.02 * index,
                    "holdout_accuracy": 0.5 - 0.01 * index,
                    "nll_delta_vs_intact": 0.02 * index,
                    "accuracy_delta_vs_intact": -0.01 * index,
                    "mean_active_mechanisms_per_token": 1,
                    "mean_writers_per_token": 0.75,
                    "mean_recipients_per_token": 1,
                    "mechanism_load": "4;4;4;4",
                    "role_mechanism_load": role_text(seed, 4, True),
                    "condition": "mediation_aux_annealed",
                })
    return raw, causal


def run_aggregate(script: Path, raw: Path, causal: Path, out: Path) -> None:
    subprocess.run([
        sys.executable, str(script), "--raw-dir", str(raw),
        "--causal-dir", str(causal), "--output-dir", str(out),
        "--seed-start", "0", "--seed-count", "2",
        "--bootstrap-replicates", "1000",
    ], check=True)


def directory_bytes(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(path.iterdir()) if item.is_file()}


def main() -> None:
    source = Path(sys.argv[1])
    script = source / "scripts" / "aggregate_phase4.py"
    with tempfile.TemporaryDirectory(prefix="sgw-phase4-test-") as temp:
        root = Path(temp)
        raw, causal = write_fixture(root)
        first = root / "first"
        second = root / "second"
        run_aggregate(script, raw, causal, first)
        run_aggregate(script, raw, causal, second)
        assert directory_bytes(first) == directory_bytes(second)
        assert len(list(csv.DictReader((first / "phase4_runs.csv").open()))) == 8
        assert len(list(csv.DictReader((first / "phase4_causal.csv").open()))) == 20
        routing = list(csv.DictReader((first / "phase4_routing.csv").open()))
        assert len(routing) == 4
        assert {row["condition"] for row in routing} == set(CONDITIONS)
        paired = (first / "phase4_paired.csv").read_text(encoding="utf-8")
        assert "mediation_aux_vs_final" in paired
        assert "mediation_aux_vs_full" in paired
        causal_text = (first / "phase4_causal_comparisons.csv").read_text(encoding="utf-8")
        assert "no_workspace_writes" in causal_text
        assert "zero_reader_inbox" in causal_text
        assert b"\r" not in b"".join(directory_bytes(first).values())

        stale = raw / "core_full_content_seed99.csv"
        shutil.copy(raw / "core_full_content_seed0.csv", stale)
        failed = subprocess.run([
            sys.executable, str(script), "--raw-dir", str(raw),
            "--causal-dir", str(causal), "--output-dir", str(root / "bad"),
            "--seed-start", "0", "--seed-count", "2",
            "--bootstrap-replicates", "10",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert failed.returncode != 0


if __name__ == "__main__":
    main()
