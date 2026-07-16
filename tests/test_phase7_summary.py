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
    "structural_core_blind", "kv_fixed_position", "kv_first_free",
    "kv_hard_router", "kv_annealed_router",
)
INTERVENTIONS = (
    "intact", "no_broadcast", "no_workspace_persistence",
    "no_workspace_output", "no_spine_workspace", "no_mechanism_output",
    "workspace_disconnected", "no_workspace_writes", "zero_reader_inbox",
    "permuted_workspace_keys", "zero_query_key", "randomized_write_slots",
    "cleared_writer_assignment", "allow_write_collisions",
    "permuted_recipients",
)
FORMAL = {
    "no_mechanism_output", "permuted_workspace_keys", "zero_query_key",
    "randomized_write_slots", "cleared_writer_assignment",
    "allow_write_collisions",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def fixture(root: Path) -> tuple[Path, Path]:
    raw, causal = root / "raw", root / "causal"
    raw.mkdir(); causal.mkdir()
    params = {"structural_core_blind": 725, "kv_fixed_position": 88,
              "kv_first_free": 88, "kv_hard_router": 112,
              "kv_annealed_router": 112}
    metrics = {
        "structural_core_blind": (1.65, .18, .82, .05),
        "kv_fixed_position": (.01, 1.0, .0001, .01),
        "kv_first_free": (.01, 1.0, .0001, .01),
        "kv_hard_router": (.50, .80, .20, .10),
        "kv_annealed_router": (.02, .99, .001, .01),
    }
    for seed in range(30):
        jitter = seed * 1e-6
        for condition in CONDITIONS:
            nll, acc, brier, ece = metrics[condition]
            core = condition == "structural_core_blind"
            row = {
                "preset": condition, "seed": seed, "steps": 800,
                "batch_size": 8, "parameters": params[condition],
                "train_count": 160, "holdout_count": 160,
                "condition": condition, "training_stream_samples": 6400,
                "final_holdout_nll": nll + jitter,
                "final_holdout_accuracy": acc,
                "final_holdout_brier": brier + jitter,
                "final_holdout_ece": ece + jitter,
                "read_slot_load": "0;0;0" if core else "54;53;53",
                "write_slot_load": "0;0;0" if core else "320;320;320",
                "mean_writers_per_token": 0.0 if core else .75,
                "final_200_collision_rate": 0.0,
                "final_200_routing_entropy": 0.1,
            }
            write_csv(raw / f"{condition}_seed{seed}.csv", [row])
        causal_rows = []
        for intervention in INTERVENTIONS:
            harmful = intervention in FORMAL
            causal_rows.append({
                "seed": seed, "intervention": intervention,
                "condition": "kv_annealed_router",
                "holdout_nll": 1.5 + jitter if harmful else .02 + jitter,
                "holdout_accuracy": .20 if harmful else .99,
                "read_slot_load": "54;53;53", "write_slot_load": "320;320;320",
            })
        write_csv(causal / f"causal_seed{seed}.csv", causal_rows)
    return raw, causal


def run(script: Path, raw: Path, causal: Path, out: Path, check: bool = True):
    return subprocess.run([
        sys.executable, str(script), "--raw-dir", str(raw),
        "--causal-dir", str(causal), "--output-dir", str(out),
        "--seed-start", "0", "--seed-count", "30",
        "--bootstrap-replicates", "1000",
    ], check=check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def files(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(path.iterdir()) if item.is_file()}


def main() -> None:
    source = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="phase7-") as temp:
        root = Path(temp); raw, causal = fixture(root)
        first, second = root / "first", root / "second"
        run(source / "scripts/aggregate_phase7.py", raw, causal, first)
        run(source / "scripts/aggregate_phase7.py", raw, causal, second)
        assert files(first) == files(second)
        assert len(list(csv.DictReader((first / "phase7_runs.csv").open()))) == 150
        assert len(list(csv.DictReader((first / "phase7_causal.csv").open()))) == 450
        manifest = json.loads((first / "phase7_manifest.json").read_text())
        assert manifest["first_free_gate_pass"] is True
        assert manifest["annealed_structural_gate_pass"] is True
        assert manifest["collision_gate_pass"] is True
        assert manifest["causal_gate_pass"] is True
        assert manifest["overall_gate_pass"] is True
        assert b"\r" not in b"".join(files(first).values())

        shutil.copy(raw / "kv_first_free_seed0.csv", raw / "extra.csv")
        assert run(source / "scripts/aggregate_phase7.py", raw, causal,
                   root / "bad", check=False).returncode != 0


if __name__ == "__main__":
    main()
