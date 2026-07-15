# SGW-ESM Phase 3 Credit Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add broadcast-forced SGW topology, temporary aligned workspace supervision, complete pathway interventions, role-conditioned route diagnostics, and a 30-seed formal experiment.

**Architecture:** Preserve the Phase 2 scalar-autodiff model. Add explicit primary-path flags to model presets, expose workspace-only auxiliary logits, and compute an annealed auxiliary loss during training. Extend deterministic evaluation traces with token roles and aggregate role-routing statistics.

**Tech Stack:** C++20, CMake 3.20+, standard library, Python 3 standard library, Bash, GitHub Actions.

## Global Constraints

- Runtime remains standard-library-only.
- Fixed-seed outputs remain byte deterministic.
- Hard routing budgets remain exact.
- Auxiliary weight reaches exactly zero at step 200 of a 400-step formal run.
- Evaluation uses primary logits only.
- Long formal experiments run in GitHub Actions.

---

### Task 1: Primary pathway flags and presets

**Files:**
- Modify: `include/sgw/config.hpp`
- Modify: `include/sgw/presets.hpp`
- Modify: `src/presets.cpp`
- Modify: `src/model.cpp`
- Test: `tests/test_presets.cpp`
- Test: `tests/test_model_forward.cpp`

**Interfaces:**
- Produces: `spine_reads_workspace`, `output_reads_workspace`, `output_reads_mechanism` in `ModelConfig`.
- Produces: `sgw_redundant` and `sgw_broadcast_forced` presets.

- [ ] Write failing tests for preset flags, parameter counts, and inference MAdd accounting.
- [ ] Verify failure.
- [ ] Implement flags without changing Phase 2 default behavior.
- [ ] Verify all existing tests and new tests pass.
- [ ] Commit.

### Task 2: Workspace auxiliary logits and annealed loss

**Files:**
- Modify: `include/sgw/model.hpp`
- Modify: `src/model.cpp`
- Modify: `include/sgw/experiment.hpp`
- Modify: `src/experiment.cpp`
- Test: `tests/test_training.cpp`
- Test: `tests/test_gradients.cpp`

**Interfaces:**
- `SequenceResult::workspace_aux_logits` contains workspace-only target logits for SGW presets.
- `TrainingConfig::workspace_aux_weight` and `workspace_aux_anneal_steps` define the schedule.
- `TrainingHistory::workspace_aux_weight` records the exact schedule.

- [ ] Write failing tests proving auxiliary logits exist, use no spine/mechanism term, and receive finite gradients.
- [ ] Write a failing schedule test proving weight is 0.5 at step 0 and exactly zero from step 200 onward.
- [ ] Implement auxiliary logits and total loss.
- [ ] Verify primary evaluation ignores auxiliary logits.
- [ ] Commit.

### Task 3: Complete pathway interventions

**Files:**
- Modify: `include/sgw/model.hpp`
- Modify: `src/model.cpp`
- Test: `tests/test_interventions.cpp`

**Interfaces:**
- Adds: `no_spine_workspace`, `no_mechanism_output`, `workspace_disconnected`.

- [ ] Write failing state/output invariance tests for each intervention.
- [ ] Implement intervention composition in one central set of effective-path booleans.
- [ ] Verify intact execution remains byte identical.
- [ ] Commit.

### Task 4: Role-conditioned route telemetry

**Files:**
- Modify: `include/sgw/dataset.hpp`
- Modify: `src/dataset.cpp`
- Modify: `include/sgw/experiment.hpp`
- Modify: `src/experiment.cpp`
- Modify: `apps/sgw_train.cpp`
- Test: `tests/test_dataset.cpp`
- Test: `tests/test_cli.sh`

**Interfaces:**
- Adds `TokenRole` and `BindingSample::roles` aligned one-to-one with tokens.
- `EvaluationMetrics::role_mechanism_load` is a flattened role-by-mechanism matrix.
- CSV encodes the matrix deterministically using `|` between roles and `;` between mechanisms.

- [ ] Write failing role alignment and exact-count tests.
- [ ] Implement role assignment in the generator.
- [ ] Aggregate active mechanisms by role during evaluation.
- [ ] Emit deterministic CSV telemetry.
- [ ] Commit.

### Task 5: Phase 3 CLI conditions and formal aggregation

**Files:**
- Modify: `include/sgw/presets.hpp`
- Modify: `apps/sgw_train.cpp`
- Create: `scripts/run_phase3_shard.sh`
- Create: `scripts/aggregate_phase3.py`
- Modify: `tests/test_cli.sh`
- Create: `tests/test_phase3_summary.sh`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Conditions: `core_small`, `core_compute_matched`, `core_param_matched`, `sgw_redundant_final_only`, `sgw_broadcast_forced_final_only`, `sgw_broadcast_forced_aux_annealed`.
- Outputs consolidated performance, causal comparisons, anneal-boundary diagnostics, and route-information statistics.

- [ ] Add failing deterministic fixture tests with strict Cartesian-product validation.
- [ ] Implement condition parsing and auxiliary schedule selection.
- [ ] Implement fixed-seed bootstrap aggregation and route information statistics.
- [ ] Reject missing, duplicate, or stale shards.
- [ ] Commit.

### Task 6: GitHub Actions formal experiment and documentation

**Files:**
- Create: `.github/workflows/phase3-experiments.yml`
- Modify: `README.md`
- Create: `results/PHASE3_PROTOCOL.md`

**Interfaces:**
- Six five-seed Actions shards followed by deterministic aggregation and SHA-256 verification.

- [ ] Add the manual formal workflow.
- [ ] Run GCC, Clang, sanitizer, and release gates locally.
- [ ] Run a two-seed Phase 3 smoke matrix.
- [ ] Publish the implementation to a new remote research branch.
- [ ] Run the 30-seed Actions matrix and inspect the artifact before making claims.
- [ ] Commit verified results and update the scientific boundary.
