# SGW-ESM Phase 8 Retention and Eviction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and formally evaluate scarce-workspace retention, skip, and eviction policies on a context-conditioned six-binding task.

**Architecture:** Extend the existing identity-preserving Key–Value path with an optional retention policy and one leading context token. Preserve tied entity/value codebooks and top-1 retrieval. Implement deterministic full-capacity, oracle, FIFO, and reservoir controls plus hard and annealed learned action routing over skip and three write/evict actions.

**Tech Stack:** C++20, CMake, Ninja, Python standard library, GitHub Actions, CPU only.

## Global Constraints

- Preserve all Phase 1–7 behavior, parameter counts, CSV schemas, and result files.
- Runtime remains standard-library-only and deterministic for fixed seeds.
- Context tokens are appended after the existing query token.
- Formal training consumes exactly `steps * batch_size` no-replacement samples.
- Learned inference actions are exact hard one-hot choices.
- The annealed condition uses hard-only routing for its final 300 updates.
- Formal CSV files use deterministic ordering and LF endings.

---

### Task 1: Add Phase 8 task and vocabulary support

**Files:** `include/sgw/dataset.hpp`, `src/dataset.cpp`, `tests/test_dataset.cpp`

- [ ] Add `TokenRole::context`, context token vocabulary methods, `RetentionTaskConfig`, `RetentionBindingStream`, and `make_retention_binding_split`.
- [ ] Generate all three relevant entities plus three irrelevant entities per episode, randomize pair order, and query only a relevant entity.
- [ ] Preserve structural train/holdout value isolation and exact stream non-repetition.
- [ ] Test sequence shape, context/relevance rule, deterministic order, metadata, and 9,600-sample stream uniqueness.
- [ ] Run focused tests and commit.

### Task 2: Add retention configuration and presets

**Files:** `include/sgw/config.hpp`, `include/sgw/presets.hpp`, `src/config.cpp`, `src/presets.cpp`, `tests/test_config_optimizer.cpp`, `tests/test_presets.cpp`

- [ ] Add `KeyValueRetentionMode` values for none, full capacity, oracle, FIFO, reservoir, hard learned, and annealed learned.
- [ ] Add six Phase 8 conditions and presets.
- [ ] Validate three-slot scarce policies, six-slot full-capacity control, context count, temperature schedule, and hard-only boundary.
- [ ] Allocate context codebook and one age-weight scalar only for learned retention policies.
- [ ] Lock exact parameter counts and estimated MAdds in tests.
- [ ] Run focused tests and commit.

### Task 3: Implement deterministic retention baselines

**Files:** `include/sgw/model.hpp`, `src/model.cpp`, `tests/test_model_forward.cpp`, `tests/test_interventions.cpp`

- [ ] Parse the leading context token and retain existing Phase 6–7 sequence handling when retention mode is none.
- [ ] Implement full-capacity, oracle, FIFO, and deterministic reservoir actions.
- [ ] Replace slot key and value atomically, reset age, and increment all other occupied ages.
- [ ] Verify exact skip/write/eviction behavior, tie-breaking, capacity bounds, and query retrieval.
- [ ] Run focused tests and commit.

### Task 4: Implement hard and annealed learned retention

**Files:** `include/sgw/model.hpp`, `src/model.cpp`, `tests/test_gradients.cpp`, `tests/test_training.cpp`

- [ ] Add context codebook and learned age coefficient.
- [ ] Score skip and all slot actions from context relevance, incoming relevance, stored relevance, and age.
- [ ] Add stable hard argmax and annealed straight-through action gates.
- [ ] Carry the entity action to the following value token and support differentiable slot replacement.
- [ ] Verify nonzero retention-router gradients during annealing and no surrogate after update 899.
- [ ] Run focused tests and commit.

### Task 5: Add retention telemetry and interventions

**Files:** `include/sgw/model.hpp`, `include/sgw/experiment.hpp`, `src/model.cpp`, `src/experiment.cpp`, `tests/test_interventions.cpp`, `tests/test_training.cpp`

- [ ] Record skip, write, eviction, relevant eviction, queried retention, read hit, entropy, disagreement, age, and per-slot loads.
- [ ] Add context-zeroing, action randomization, forced FIFO, forced relevant eviction, context permutation, and skip-disable interventions.
- [ ] Verify each intervention changes only its intended path and degrades an oracle-compatible sample.
- [ ] Run focused tests and commit.

### Task 6: Expose Phase 8 CLI and deterministic experiment shards

**Files:** `apps/sgw_train.cpp`, `scripts/run_phase8_shard.sh`, `tests/test_cli.sh`

- [ ] Parse all six conditions and select the retention task automatically.
- [ ] Emit task identity, policy schedule, retention telemetry, final-300 metrics, and causal rows.
- [ ] Run six paired conditions per seed with 1200 updates and batch size 8.
- [ ] Verify repeated CLI runs are byte-identical and consume exactly 9,600 online samples.
- [ ] Run focused tests and commit.

### Task 7: Add strict Phase 8 aggregation and Actions matrix

**Files:** `scripts/aggregate_phase8.py`, `tests/test_phase8_summary.py`, `CMakeLists.txt`, `.github/workflows/phase8-experiments.yml`, `.gitignore`

- [ ] Enforce the complete condition-by-seed Cartesian product and reject stale or malformed files.
- [ ] Compute 50,000-replicate paired intervals for learned-vs-FIFO, learned-vs-reservoir, hard-vs-annealed, telemetry, and interventions.
- [ ] Emit runs, summary, paired, retention, causal-comparison, manifest, and SHA-256 files.
- [ ] Add six five-seed Actions shards and deterministic aggregation.
- [ ] Run synthetic aggregation tests and a two-seed smoke matrix, then commit.

### Task 8: Validate, execute, and report

**Files:** `results/PHASE8_VALIDATION.md`, `results/phase8-formal/*`

- [ ] Pass GCC Debug, Clang Debug, GCC Release, and ASan/UBSan gates.
- [ ] Publish `agent/phase8-retention-eviction` and open a draft PR against the Phase 7 branch.
- [ ] Run the 30-seed formal Actions matrix.
- [ ] Download, verify, and independently reaggregate the artifact.
- [ ] Commit the verified result tables and a bounded scientific report.
