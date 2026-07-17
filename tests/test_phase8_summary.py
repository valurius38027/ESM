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
    "kv_full_capacity", "kv_oracle_retention", "kv_fifo_eviction",
    "kv_reservoir", "kv_hard_retention", "kv_annealed_retention",
)
INTERVENTIONS = (
    "intact", "zero_query_context", "randomized_retention_actions",
    "force_fifo_retention", "force_relevant_eviction",
    "permuted_context_labels", "disable_retention_skip",
    "no_workspace_writes", "no_workspace_persistence",
    "no_mechanism_output",
)
FORMAL = set(INTERVENTIONS) - {"intact", "force_fifo_retention"}


def write_csv(path: Path, data: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def fixture(root: Path) -> tuple[Path, Path]:
    raw, causal = root / "raw", root / "causal"
    raw.mkdir(); causal.mkdir()
    params = {
        "kv_full_capacity": 120, "kv_oracle_retention": 120,
        "kv_fifo_eviction": 120, "kv_reservoir": 120,
        "kv_hard_retention": 145, "kv_annealed_retention": 145,
    }
    metrics = {
        "kv_full_capacity": (.01, 1.0, .0001, .01),
        "kv_oracle_retention": (.02, .995, .0002, .01),
        "kv_fifo_eviction": (1.10, .55, .55, .20),
        "kv_reservoir": (1.25, .45, .65, .25),
        "kv_hard_retention": (.55, .78, .24, .12),
        "kv_annealed_retention": (.08, .95, .02, .04),
    }
    for seed in range(30):
        jitter = seed * 1e-6
        for condition in CONDITIONS:
            nll, accuracy, brier, ece = metrics[condition]
            width = 6 if condition == "kv_full_capacity" else 3
            row = {
                "preset": condition, "seed": seed, "steps": 1200,
                "batch_size": 8, "parameters": params[condition],
                "train_count": 160, "holdout_count": 160,
                "sequence_length": 15, "condition": condition,
                "training_stream_samples": 9600,
                "final_holdout_nll": nll + jitter,
                "final_holdout_accuracy": accuracy,
                "final_holdout_brier": brier + jitter,
                "final_holdout_ece": ece + jitter,
                "read_slot_load": "27;27;27;27;26;26" if width == 6 else "54;53;53",
                "write_slot_load": ";".join(["160"] * width) if width == 6 else "160;160;160",
                "eviction_slot_load": ";".join(["0"] * width) if width == 6 else "80;80;80",
                "router_anneal_steps": 900 if condition == "kv_annealed_retention" else 0,
                "final_200_disagreement_rate": 0.0,
                "retention_write_rate": .5,
                "retention_skip_rate": .5,
                "retention_eviction_rate": .25,
                "relevant_eviction_rate": .02 if condition == "kv_annealed_retention" else .25,
                "queried_entity_retention_rate": .96 if condition == "kv_annealed_retention" else .60,
                "query_read_hit_rate": .95 if condition == "kv_annealed_retention" else .60,
                "mean_retained_age": 2.0,
                "final_300_retention_write_rate": .5,
                "final_300_retention_skip_rate": .5,
                "final_300_retention_eviction_rate": .25,
                "final_300_relevant_eviction_rate": .02,
                "final_300_query_read_hit_rate": .95,
            }
            write_csv(raw / f"{condition}_seed{seed}.csv", [row])
        causal_rows = []
        for intervention in INTERVENTIONS:
            harmful = intervention in FORMAL
            causal_rows.append({
                "seed": seed, "intervention": intervention,
                "condition": "kv_annealed_retention",
                "holdout_nll": 1.0 + jitter if harmful else .08 + jitter,
                "holdout_accuracy": .30 if harmful else .95,
                "read_slot_load": "54;53;53", "write_slot_load": "160;160;160",
                "eviction_slot_load": "80;80;80",
                "retention_write_rate": .5, "retention_skip_rate": .5,
                "retention_eviction_rate": .25, "relevant_eviction_rate": .02,
                "queried_entity_retention_rate": .95, "query_read_hit_rate": .95,
                "mean_retained_age": 2.0,
            })
        write_csv(causal / f"causal_seed{seed}.csv", causal_rows)
    return raw, causal


def run(script: Path, raw: Path, causal: Path, out: Path, check: bool = True,
        allow_short: bool = False):
    command = [
        sys.executable, str(script), "--raw-dir", str(raw),
        "--causal-dir", str(causal), "--output-dir", str(out),
        "--seed-start", "0", "--seed-count", "30",
        "--bootstrap-replicates", "1000",
    ]
    if allow_short:
        command.append("--allow-short-run")
    return subprocess.run(command, check=check, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)


def files(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(path.iterdir()) if item.is_file()}


def main() -> None:
    source = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="phase8-") as temp:
        root = Path(temp); raw, causal = fixture(root)
        first, second = root / "first", root / "second"
        script = source / "scripts/aggregate_phase8.py"
        run(script, raw, causal, first)
        run(script, raw, causal, second)
        assert files(first) == files(second)
        assert len(list(csv.DictReader((first / "phase8_runs.csv").open()))) == 180
        assert len(list(csv.DictReader((first / "phase8_causal.csv").open()))) == 300
        manifest = json.loads((first / "phase8_manifest.json").read_text())
        for gate in (
            "oracle_capacity_gate_pass", "annealed_accuracy_gate_pass",
            "annealed_vs_fifo_gate_pass", "annealed_vs_reservoir_gate_pass",
            "retention_quality_gate_pass", "hard_only_window_gate_pass",
            "causal_gate_pass", "overall_gate_pass",
        ):
            assert manifest[gate] is True
        assert b"\r" not in b"".join(files(first).values())

        # The formal aggregator rejects short runs unless the explicit smoke flag is used.
        for path in raw.glob("*.csv"):
            current = list(csv.DictReader(path.open()))
            current[0]["steps"] = "4"
            current[0]["training_stream_samples"] = "32"
            write_csv(path, current)
        assert run(script, raw, causal, root / "short-bad", check=False).returncode != 0
        run(script, raw, causal, root / "short-ok", allow_short=True)

        shutil.copy(raw / "kv_fifo_eviction_seed0.csv", raw / "extra.csv")
        assert run(script, raw, causal, root / "bad", check=False).returncode != 0


if __name__ == "__main__":
    main()
