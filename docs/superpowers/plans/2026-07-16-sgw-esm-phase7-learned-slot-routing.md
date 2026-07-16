# SGW-ESM Phase 7 Learned Slot Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and formally evaluate position-independent first-free, hard learned, and annealed learned sparse workspace write routing.

**Architecture:** Extend the Phase 6 KV path with an explicit write-routing mode and slot codebook. Preserve hard one-slot forward writes; use a straight-through soft surrogate only during the annealed training window. Randomize binding-pair order at the data layer and expose routing telemetry and causal interventions through the existing deterministic experiment pipeline.

**Tech Stack:** C++20, CMake, Ninja, Python standard library, GitHub Actions, CPU only.

## Global Constraints

- Preserve all Phase 1–6 behavior and result files.
- Runtime remains standard-library-only.
- Formal CSVs use LF endings and deterministic ordering.
- Training consumes exactly `steps * batch_size` no-replacement structural samples.
- Final 200 updates of the annealed condition use hard-only routing.
- Every inference write and read is top-1.

---

### Task 1: Add Phase 7 configuration and presets

**Files:** `include/sgw/config.hpp`, `include/sgw/presets.hpp`, `src/config.cpp`, `src/presets.cpp`, `tests/test_config_optimizer.cpp`, `tests/test_presets.cpp`

- [ ] Add `KeyValueWriteRoutingMode` with fixed-position, first-free, hard-learned, and annealed-learned values.
- [ ] Add four Phase 7 conditions and presets.
- [ ] Validate router dimensions, temperature bounds, and anneal boundary.
- [ ] Verify parameter counts: first-free adds zero router parameters; learned modes add `workspace_slots * key_dim`.
- [ ] Run focused tests and commit.

### Task 2: Randomize structural binding order

**Files:** `include/sgw/dataset.hpp`, `src/dataset.cpp`, `tests/test_dataset.cpp`

- [ ] Add a structural sample-order mode that deterministically permutes complete entity-value pairs.
- [ ] Prove target metadata follows the moved pair.
- [ ] Prove train/holdout pair isolation and stream uniqueness remain intact.
- [ ] Run focused tests and commit.

### Task 3: Implement first-free and learned write routing

**Files:** `include/sgw/model.hpp`, `src/model.cpp`, `tests/test_model_forward.cpp`, `tests/test_gradients.cpp`

- [ ] Add slot-router codebook allocation for learned modes.
- [ ] Add stable available-slot scoring and first-free routing.
- [ ] Carry entity assignment to the following value token.
- [ ] Add straight-through annealed route weights and hard-only final window control.
- [ ] Verify one write per binding, no collisions under intact routing, nonzero router gradients during annealing, and zero surrogate after annealing.
- [ ] Run focused tests and commit.

### Task 4: Add interventions and routing telemetry

**Files:** `include/sgw/model.hpp`, `include/sgw/experiment.hpp`, `src/model.cpp`, `src/experiment.cpp`, `tests/test_interventions.cpp`, `tests/test_training.cpp`

- [ ] Add randomized-write, cleared-assignment, and collision-enabled interventions.
- [ ] Record collision, entropy, disagreement, write load, and hard-only-window metrics.
- [ ] Verify intervention names and expected causal path changes.
- [ ] Run focused tests and commit.

### Task 5: Expose Phase 7 CLI conditions

**Files:** `apps/sgw_train.cpp`, `tests/test_cli.sh`

- [ ] Parse four conditions and emit schedule and routing telemetry.
- [ ] Emit causal rows for the three learned/first-free KV variants.
- [ ] Verify byte-identical repeated runs and exact online sample accounting.
- [ ] Run CLI gate and commit.

### Task 6: Add formal aggregation and Actions matrix

**Files:** `scripts/run_phase7_shard.sh`, `scripts/aggregate_phase7.py`, `tests/test_phase7_summary.py`, `CMakeLists.txt`, `.github/workflows/phase7-experiments.yml`, `.gitignore`

- [ ] Enforce the full condition × seed Cartesian product and reject stale files.
- [ ] Compute paired bootstrap intervals for performance, routing, and interventions.
- [ ] Add six 5-seed shards, 800 updates, batch size 8, 50,000 bootstrap replicates.
- [ ] Verify LF-only deterministic outputs and SHA-256 manifest.
- [ ] Run smoke matrix and commit.

### Task 7: Validate and execute

**Files:** `results/PHASE7_VALIDATION.md`, `results/phase7-formal/*`

- [ ] Pass GCC Debug, Clang Debug, GCC Release, and ASan/UBSan gates.
- [ ] Push `feature/phase7-learned-slot-routing` to `valurius38027/ESM` and open a draft PR.
- [ ] Run the 30-seed formal matrix in Actions.
- [ ] Download and independently reaggregate the artifact.
- [ ] Commit verified results and a bounded scientific report.
