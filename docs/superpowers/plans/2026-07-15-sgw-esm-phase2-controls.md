# SGW-ESM Phase 2 Experimental Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add parameter/compute-matched recurrent controls, deterministic post-training workspace interventions, and a 30-seed GitHub Actions experiment pipeline.

**Architecture:** Preserve the Phase 1 model and scalar autodiff. Add explicit model presets and evaluation-only forward interventions, then extend the CLI and aggregation pipeline so one SGW training run emits all causal evaluations. Use sharded Actions jobs for formal experiments.

**Tech Stack:** C++20, CMake 3.20+, standard library, Python 3 standard library, Bash, GitHub Actions.

## Global Constraints

- Runtime remains standard-library-only.
- All model and experiment behavior remains deterministic for a fixed seed.
- Hard routing budgets remain exact.
- Formal runs use 30 paired seeds, 400 steps, batch size 8, 160 train samples, and 48 holdout samples.
- Every host-side local command remains bounded by 110 seconds; long experiments execute in GitHub Actions.

---

### Task 1: Model presets and exact matching

**Files:**
- Create: `include/sgw/presets.hpp`
- Create: `src/presets.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_presets.cpp`

**Interfaces:**
- Produces: `enum class ModelPreset`, `parse_model_preset`, `model_preset_name`, and `make_model_config`.

- [ ] Write tests asserting exact parameter and MAdd counts for all three presets.
- [ ] Verify tests fail because preset APIs do not exist.
- [ ] Implement preset construction and parsing.
- [ ] Verify all tests pass under GCC and Clang.
- [ ] Commit.

### Task 2: Evaluation-only causal interventions

**Files:**
- Modify: `include/sgw/model.hpp`
- Modify: `src/model.cpp`
- Modify: `include/sgw/experiment.hpp`
- Modify: `src/experiment.cpp`
- Test: `tests/test_interventions.cpp`

**Interfaces:**
- Produces: `ForwardIntervention`, `forward_intervention_name`, and an intervention argument on `forward_sequence` and `evaluate`.

- [ ] Write tests for no-broadcast mutation, zeroed cross-token workspace persistence, disabled output projection, and cyclic recipient permutation.
- [ ] Verify tests fail on missing APIs.
- [ ] Implement the minimal intervention points without changing normal execution.
- [ ] Verify Phase 1 tests and new intervention tests pass.
- [ ] Commit.

### Task 3: Phase 2 CLI output

**Files:**
- Modify: `apps/sgw_train.cpp`
- Modify: `tests/test_cli.sh`
- Test: `tests/test_training.cpp`

**Interfaces:**
- Consumes: model presets and interventions.
- Produces: one primary run CSV and optional causal CSV containing five rows from the same trained SGW model.

- [ ] Add failing CLI tests for `core_small`, `core_compute_matched`, `core_param_matched`, `sgw`, and deterministic causal output.
- [ ] Implement `--preset` and `--causal-output` while retaining `--mode` aliases for Phase 1 compatibility.
- [ ] Add elapsed time and exact budget fields.
- [ ] Verify byte-identical repeated outputs except for elapsed time, which is excluded from deterministic comparison artifacts.
- [ ] Commit.

### Task 4: Deterministic formal aggregation

**Files:**
- Replace: `scripts/summarize_runs.py`
- Create: `scripts/run_phase2_shard.sh`
- Create: `scripts/aggregate_phase2.py`
- Modify: `tests/test_summary.sh`

**Interfaces:**
- Produces: `phase2_runs.csv`, `phase2_summary.csv`, `phase2_paired.csv`, `phase2_causal.csv`, `phase2_manifest.json`, and `SHA256SUMS`.

- [ ] Write failing fixture tests for complete Cartesian seed/preset coverage and causal rows.
- [ ] Implement strict validation, fixed-seed paired bootstrap intervals, direction counts, and deterministic output.
- [ ] Verify stale or duplicate shards are rejected.
- [ ] Commit.

### Task 5: GitHub Actions CI and formal experiments

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/phase2-experiments.yml`
- Modify: `README.md`

**Interfaces:**
- CI builds GCC, Clang, sanitizer, and release gates.
- Experiment workflow runs six five-seed shards, aggregates them, and uploads raw and consolidated artifacts.

- [ ] Add workflow syntax tests where locally possible.
- [ ] Add build/test matrix and sharded formal workflow.
- [ ] Document exact reproduction commands and scientific gates.
- [ ] Run local full validation.
- [ ] Commit and publish to `valurius38027/ESM` on a research branch.
