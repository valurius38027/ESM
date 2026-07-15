#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONDITIONS = (
    "core_small",
    "core_compute_matched",
    "core_param_matched",
    "sgw_redundant_final_only",
    "sgw_broadcast_forced_final_only",
    "sgw_broadcast_forced_aux_annealed",
)
INTERVENTIONS = (
    "intact",
    "no_broadcast",
    "no_workspace_persistence",
    "no_workspace_output",
    "no_spine_workspace",
    "no_mechanism_output",
    "workspace_disconnected",
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
    "last_workspace_aux_window_loss"
).split(",")
CAUSAL_FIELDS = (
    "seed,intervention,holdout_nll,holdout_accuracy,nll_delta_vs_intact,"
    "accuracy_delta_vs_intact,mean_active_mechanisms_per_token,"
    "mean_writers_per_token,mean_recipients_per_token,mechanism_load,"
    "role_mechanism_load,condition"
).split(",")


def role_text(seed: int) -> str:
    rows = []
    names = ("entity", "value", "filler", "query_marker", "query_entity")
    for role, name in enumerate(names):
        counts = [1 + ((seed + role + mechanism) % 4) for mechanism in range(6)]
        rows.append(name + ":" + ";".join(str(value) for value in counts))
    return "|".join(rows)


def write_fixture(root: Path) -> tuple[Path, Path]:
    raw = root / "raw"
    causal = root / "causal"
    raw.mkdir()
    causal.mkdir()
    for seed in range(2):
        for index, condition in enumerate(CONDITIONS):
            preset = condition
            if condition == "sgw_redundant_final_only":
                preset = "sgw"
            elif condition.startswith("sgw_broadcast_forced"):
                preset = "sgw_broadcast_forced"
            params = (327, 2232, 2181, 2182, 2182, 2182)[index]
            madds = (210, 1092, 1474, 1092, 1002, 1002)[index]
            nll = 1.7 - 0.08 * index + 0.03 * seed
            aux_weight = 0.5 if condition.endswith("aux_annealed") else 0.0
            aux_steps = 200 if aux_weight else 0
            row = {
                "preset": preset,
                "seed": seed,
                "steps": 400,
                "batch_size": 8,
                "parameters": params,
                "train_count": 160,
                "holdout_count": 48,
                "sequence_length": 10,
                "chance_nll": 1.609,
                "initial_train_nll": 1.7,
                "initial_holdout_nll": 1.7,
                "final_train_nll": 0.4 - 0.02 * index,
                "final_holdout_nll": nll,
                "final_train_accuracy": 0.8,
                "final_holdout_accuracy": 0.4 + 0.02 * index,
                "mean_estimated_madds_per_token": madds,
                "mean_active_mechanisms_per_token": 0 if index < 3 else 2,
                "mean_writers_per_token": 0 if index < 3 else 1,
                "mean_recipients_per_token": 0 if index < 3 else 2,
                "first_window_loss": 1.6,
                "last_window_loss": 0.5,
                "mechanism_load": "0;0;0;0;0;0" if index < 3 else "10;11;12;13;14;15",
                "role_mechanism_load": (
                    "entity:0;0;0;0;0;0|value:0;0;0;0;0;0|"
                    "filler:0;0;0;0;0;0|query_marker:0;0;0;0;0;0|"
                    "query_entity:0;0;0;0;0;0"
                    if index < 3 else role_text(seed)
                ),
                "condition": condition,
                "workspace_aux_initial_weight": aux_weight,
                "workspace_aux_anneal_steps": aux_steps,
                "first_primary_window_loss": 1.6,
                "last_primary_window_loss": 0.5,
                "last_workspace_aux_window_loss": 0.6,
            }
            path = raw / f"{condition}_seed{seed}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerow(row)

        path = causal / f"causal_seed{seed}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
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
                    "mean_active_mechanisms_per_token": 2,
                    "mean_writers_per_token": 1,
                    "mean_recipients_per_token": 2,
                    "mechanism_load": "10;11;12;13;14;15",
                    "role_mechanism_load": role_text(seed),
                    "condition": "sgw_broadcast_forced_aux_annealed",
                })
    return raw, causal


def run_aggregate(script: Path, raw: Path, causal: Path, out: Path) -> None:
    subprocess.run(
        [sys.executable, str(script), "--raw-dir", str(raw),
         "--causal-dir", str(causal), "--output-dir", str(out),
         "--seed-start", "0", "--seed-count", "2",
         "--bootstrap-replicates", "1000"],
        check=True,
    )


def directory_bytes(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(path.iterdir()) if item.is_file()}


def main() -> None:
    source = Path(sys.argv[1])
    script = source / "scripts" / "aggregate_phase3.py"
    with tempfile.TemporaryDirectory(prefix="sgw-phase3-test-") as temp:
        root = Path(temp)
        raw, causal = write_fixture(root)
        first = root / "first"
        second = root / "second"
        run_aggregate(script, raw, causal, first)
        run_aggregate(script, raw, causal, second)
        assert directory_bytes(first) == directory_bytes(second)
        assert len(list(csv.DictReader((first / "phase3_runs.csv").open()))) == 12
        assert len(list(csv.DictReader((first / "phase3_causal.csv").open()))) == 16
        routing = list(csv.DictReader((first / "phase3_routing.csv").open()))
        assert len(routing) == 6
        assert {row["condition"] for row in routing} == set(CONDITIONS)
        aux = next(row for row in routing if row["condition"].endswith("aux_annealed"))
        assert 0.0 <= float(aux["normalized_mechanism_entropy_mean"]) <= 1.0
        assert float(aux["role_mechanism_mutual_information_mean"]) >= 0.0
        paired = (first / "phase3_paired.csv").read_text(encoding="utf-8")
        assert "forced_aux_vs_forced_final" in paired
        assert "core_compute_matched_vs_forced_aux" in paired
        causal_text = (first / "phase3_causal_comparisons.csv").read_text(encoding="utf-8")
        assert "workspace_disconnected" in causal_text
        assert b"\r" not in b"".join(directory_bytes(first).values())

        stale = raw / "core_small_seed99.csv"
        shutil.copy(raw / "core_small_seed0.csv", stale)
        failed = subprocess.run(
            [sys.executable, str(script), "--raw-dir", str(raw),
             "--causal-dir", str(causal), "--output-dir", str(root / "bad"),
             "--seed-start", "0", "--seed-count", "2",
             "--bootstrap-replicates", "10"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert failed.returncode != 0


if __name__ == "__main__":
    main()
