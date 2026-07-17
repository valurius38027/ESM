#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

CONDITIONS = (
    "kv_delayed_oracle", "kv_delayed_fifo", "kv_delayed_reservoir",
    "kv_delayed_hard", "kv_delayed_annealed_direct",
    "kv_delayed_annealed_curriculum",
)
INTERVENTIONS = (
    "intact", "zero_query_context", "randomized_retention_actions",
    "force_fifo_retention", "force_relevant_eviction",
    "permuted_context_labels", "disable_retention_skip",
    "no_workspace_writes", "no_workspace_persistence", "no_mechanism_output",
    "remove_delay_distractors", "relevant_looking_delay_distractors",
    "reverse_delay_block",
)


def write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, raw: Path, causal: Path, out: Path, expect: int = 0) -> None:
    result = subprocess.run([
        sys.executable, str(root / "scripts/aggregate_phase9.py"),
        "--raw-dir", str(raw), "--causal-dir", str(causal),
        "--output-dir", str(out), "--seed-start", "0", "--seed-count", "30",
        "--bootstrap-replicates", "100",
    ], check=False, capture_output=True, text=True)
    if result.returncode != expect:
        raise AssertionError(result.stdout + result.stderr)


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        raw, causal, output = base / "raw", base / "causal", base / "output"
        run_fields = [
            "seed", "steps", "batch_size", "parameters", "train_count",
            "holdout_count", "sequence_length", "condition",
            "training_stream_samples", "read_slot_load", "write_slot_load",
            "eviction_slot_load", "router_anneal_steps",
            "final_450_disagreement_rate", "relevant_eviction_rate",
            "query_read_hit_rate", "distractor_write_rate",
            "distractor_eviction_rate", "relevant_survival_rate",
            "final_450_distractor_write_rate",
            "final_450_distractor_eviction_rate",
            "final_450_relevant_survival_rate", "delay_binding_count",
            "mean_source_query_distance", "final_holdout_nll",
            "final_holdout_brier", "final_holdout_ece",
            "final_holdout_accuracy",
        ] + [f"delay{delay}_{suffix}" for delay in (0, 6, 12, 18, 24)
             for suffix in ("nll", "accuracy", "query_hit", "survival")]
        profiles = {
            "kv_delayed_oracle": (0.01, 0.001, 1.0, 1.0, 1.0, 120),
            "kv_delayed_fifo": (1.4, 0.62, 0.50, 0.50, 0.45, 120),
            "kv_delayed_reservoir": (1.3, 0.58, 0.55, 0.52, 0.48, 120),
            "kv_delayed_hard": (1.35, 0.60, 0.52, 0.51, 0.47, 145),
            "kv_delayed_annealed_direct": (0.40, 0.14, 0.92, 0.91, 0.90, 145),
            "kv_delayed_annealed_curriculum": (0.20, 0.06, 0.97, 0.96, 0.96, 145),
        }
        for seed in range(30):
            for condition in CONDITIONS:
                nll, brier, accuracy, hit, survival, parameters = profiles[condition]
                row: dict[str, object] = {
                    "seed": seed, "steps": 1800, "batch_size": 8,
                    "parameters": parameters, "train_count": 160,
                    "holdout_count": 160, "sequence_length": 51,
                    "condition": condition, "training_stream_samples": 14400,
                    "read_slot_load": "54;53;53", "write_slot_load": "400;400;400",
                    "eviction_slot_load": "100;100;100", "router_anneal_steps": 1350,
                    "final_450_disagreement_rate": 0.0,
                    "relevant_eviction_rate": 0.01 if "annealed" in condition else 0.2,
                    "query_read_hit_rate": hit, "distractor_write_rate": 0.05,
                    "distractor_eviction_rate": 0.04,
                    "relevant_survival_rate": survival,
                    "final_450_distractor_write_rate": 0.03,
                    "final_450_distractor_eviction_rate": 0.02,
                    "final_450_relevant_survival_rate": survival,
                    "delay_binding_count": 18, "mean_source_query_distance": 44,
                    "final_holdout_nll": nll, "final_holdout_brier": brier,
                    "final_holdout_ece": 0.02, "final_holdout_accuracy": accuracy,
                }
                for delay in (0, 6, 12, 18, 24):
                    row[f"delay{delay}_nll"] = nll
                    row[f"delay{delay}_accuracy"] = accuracy
                    row[f"delay{delay}_query_hit"] = hit
                    row[f"delay{delay}_survival"] = survival
                write(raw / f"{condition}_seed{seed}.csv", run_fields, [row])
            causal_fields = [
                "seed", "intervention", "condition", "holdout_nll",
                "holdout_accuracy", "read_slot_load", "write_slot_load",
                "eviction_slot_load", "distractor_write_rate",
                "distractor_eviction_rate", "relevant_survival_rate",
            ]
            causal_rows = []
            for intervention in INTERVENTIONS:
                intact = intervention == "intact" or intervention in {
                    "remove_delay_distractors", "reverse_delay_block"}
                causal_rows.append({
                    "seed": seed, "intervention": intervention,
                    "condition": "kv_delayed_annealed_curriculum",
                    "holdout_nll": 0.2 if intact else 1.2,
                    "holdout_accuracy": 0.97 if intact else 0.40,
                    "read_slot_load": "54;53;53",
                    "write_slot_load": "400;400;400",
                    "eviction_slot_load": "100;100;100",
                    "distractor_write_rate": 0.05,
                    "distractor_eviction_rate": 0.04,
                    "relevant_survival_rate": 0.96 if intact else 0.3,
                })
            write(causal / f"causal_seed{seed}.csv", causal_fields, causal_rows)
        run(root, raw, causal, output)
        manifest = json.loads((output / "phase9_manifest.json").read_text())
        assert manifest["overall_gate_pass"] is True
        assert manifest["run_rows"] == 180
        assert manifest["causal_rows"] == 390
        assert (output / "SHA256SUMS").is_file()
        (raw / "extra.csv").write_text("x\n1\n", encoding="utf-8")
        run(root, raw, causal, base / "bad-output", expect=1)


if __name__ == "__main__":
    main()
