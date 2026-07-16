# SGW-ESM Phase 6 Identity-Preserving Key-Value Mediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact and tied learned sparse key-value workspace conditions that test whether identity-preserving storage and query-conditioned retrieval eliminate Phase 5 structural anti-generalization.

**Architecture:** Add a dedicated key-value mediation path inside `SgwEsmModel` rather than modifying the existing generic and fixed-mediation paths. Each binding position owns one slot with separate key/value fields; 6A uses literal identity matching and 6B uses shared normalized entity/value codebooks, deterministic top-1 retrieval, and tied decoding.

**Tech Stack:** C++20, CMake, Ninja, Python standard library, GitHub Actions, CPU only.

## Global Constraints

- Runtime remains standard-library-only and CPU-only.
- Existing Phase 1-5 behavior, parameter counts, outputs, and aggregators remain unchanged.
- Phase 6 uses the existing structural holdout and deterministic no-replacement stream.
- Formal runs use 30 paired seeds, 800 steps, batch size 8, six shards, and 50,000 bootstrap replicates.
- All new routes and interventions are deterministic and checksum-verifiable.

---

### Task 1: Add Phase 6 configuration, conditions, and invariants

**Files:**
- Modify: `include/sgw/config.hpp`
- Modify: `src/config.cpp`
- Modify: `include/sgw/presets.hpp`
- Modify: `src/presets.cpp`
- Test: `tests/test_config_optimizer.cpp`
- Test: `tests/test_presets.cpp`

**Interfaces:**
- Produces: `enum class KeyValueMediationMode { none, symbolic_exact, learned_tied };`
- Produces: `ModelConfig::key_value_mode`, `entity_count`, `value_count`, `key_dim`, `value_dim`, `key_value_logit_scale`.
- Produces conditions and presets `structural_kv_exact` and `structural_kv_learned`.

- [ ] Write failing tests asserting exact preset shape, learned preset shape, zero versus non-zero parameter intent, token-count invariants, and invalid combinations.
- [ ] Run `cmake --build build-phase6 && ./build-phase6/sgw_tests` and verify the new assertions fail before implementation.
- [ ] Add the enum, fields, parser/name mappings, condition-to-preset mappings, and validation rules.
- [ ] Configure both presets with three slots, four trace mechanisms, `key_dim=8`, `value_dim=8`, five output classes, and `key_value_logit_scale=6.0`.
- [ ] Rebuild and verify the configuration/preset tests pass.
- [ ] Commit as `feat: add phase 6 key value presets`.

### Task 2: Implement identity-preserving key-value forward paths

**Files:**
- Modify: `include/sgw/model.hpp`
- Modify: `src/model.cpp`
- Test: `tests/test_model_forward.cpp`
- Test: `tests/test_interventions.cpp`

**Interfaces:**
- Produces: `ForwardIntervention::permuted_workspace_keys` and `ForwardIntervention::zero_query_key`.
- Produces: `StepTrace::read_slots`.
- Produces private codebook pointers `kv_entity_codebook_` and `kv_value_codebook_`.
- Produces a dedicated key-value forward path selected when `key_value_mode != none`.

- [ ] Write failing exact-path tests for literal slot storage, perfect query retrieval, zero trainable scalars, deterministic traces, and write-route preservation under `no_workspace_writes`.
- [ ] Write failing learned-path tests for two codebook parameter names and shapes, normalized same-entity top-1 retrieval, tied value decoding, and deterministic repeatability.
- [ ] Write failing intervention tests showing that key permutation, zero query, suppressed writes, suppressed persistence, and suppressed output damage the learned prediction without changing intended write routes.
- [ ] Add normalized-vector and cosine helpers using `exp(0.5 * log(sum_squares + 1e-12))`; do not add a new autodiff operator.
- [ ] Implement the exact path with literal entity/value slot metadata and fixed logits.
- [ ] Implement the learned path with shared codebook rows, disjoint key/value writes, stable top-1 read, tied cosine decoding, and no bias or recurrent transforms.
- [ ] Extend route traces and MADD accounting for the new path.
- [ ] Run focused model and intervention tests, then all `sgw_tests`.
- [ ] Commit as `feat: add identity preserving key value mediation`.

### Task 3: Extend CLI, causal telemetry, and deterministic training accounting

**Files:**
- Modify: `apps/sgw_train.cpp`
- Modify: `tests/test_cli.sh`
- Modify: `tests/test_training.cpp`

**Interfaces:**
- Consumes: Phase 6 conditions and interventions from Tasks 1-2.
- Produces: CLI condition/preset names, structural-stream selection, Phase 6 causal rows, and read-slot telemetry.

- [ ] Write failing CLI tests for both Phase 6 conditions, exact `samples_consumed=steps*batch_size`, `parameter_count=0` for exact, positive parameter count for learned, and deterministic primary/causal CSV bytes.
- [ ] Write a training test confirming the exact zero-parameter condition consumes the complete unique stream without optimizer failure.
- [ ] Add Phase 6 condition handling to usage, default-condition mapping, structural dataset selection, and causal intervention enumeration.
- [ ] Add `read_slot_load` to primary/causal telemetry without changing existing column values for Phase 1-5 conditions.
- [ ] Run `tests/test_cli.sh` and all C++ tests.
- [ ] Commit as `feat: expose phase 6 experiments and telemetry`.

### Task 4: Add strict Phase 6 aggregation and regression tests

**Files:**
- Create: `scripts/aggregate_phase6.py`
- Create: `tests/test_phase6_summary.py`
- Modify: `CMakeLists.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `phase6_runs.csv`, `phase6_causal.csv`, `phase6_summary.csv`, `phase6_paired.csv`, `phase6_causal_comparisons.csv`, `phase6_routing.csv`, `phase6_manifest.json`, and `SHA256SUMS`.

- [ ] Write a synthetic regression fixture covering 30 seeds, four conditions, six causal interventions, favorable exact/learned gates, duplicate-file rejection, missing-file rejection, and sample-count rejection.
- [ ] Run the new test and verify failure because `aggregate_phase6.py` does not exist.
- [ ] Implement strict raw/causal file discovery and schema checks by adapting Phase 5 logic without importing third-party modules.
- [ ] Implement deterministic 50,000-replicate paired bootstrap comparisons for the capacity, learned structural, bounded-control, and causal gates.
- [ ] Emit manifest gate booleans and SHA-256 checksums in deterministic filename order.
- [ ] Register `sgw_phase6_summary` in CTest and run the regression test plus Phase 2-5 aggregator tests.
- [ ] Commit as `feat: add phase 6 formal aggregation`.

### Task 5: Add sharded formal experiment workflow

**Files:**
- Create: `scripts/run_phase6_shard.sh`
- Create: `.github/workflows/phase6-experiments.yml`
- Test: end-to-end local smoke output under `results/phase6-smoke/`

**Interfaces:**
- Consumes: Phase 6 CLI and aggregator.
- Produces: six deterministic shard artifacts and one consolidated formal artifact.

- [ ] Add a shard runner that executes the four paired conditions for a contiguous seed range and emits causal rows for learned key-value mediation.
- [ ] Add a six-shard manual Actions workflow with defaults `steps=800`, `seed_count=30`, Release/Ninja build, strict divisibility checks, 50,000 bootstrap replicates, checksum verification, and 90-day artifact retention.
- [ ] Run a local two-seed smoke matrix at reduced steps, aggregate it, and verify all checksums.
- [ ] Remove smoke outputs from tracking and run `git diff --check`.
- [ ] Commit as `ci: add phase 6 formal experiment matrix`.

### Task 6: Validate, publish, execute, and adjudicate

**Files:**
- Create after results: `results/PHASE6_VALIDATION.md`
- Commit after results: `results/phase6-formal/`

**Interfaces:**
- Produces: a draft Phase 6 PR, verified formal artifact, committed result tables, and bounded scientific conclusion.

- [ ] Configure and run GCC Debug, Clang Debug, GCC Release, and ASan/UBSan gates.
- [ ] Verify all C++/CLI/aggregator tests and `git diff --check` pass.
- [ ] Publish the branch and draft PR only to the ESM repository; never use an unrelated selected repository.
- [ ] Execute all six 30-seed shards and aggregate only after every shard succeeds.
- [ ] Download the formal artifact, verify the outer digest and internal `SHA256SUMS`, and independently recompute point estimates and paired interval signs.
- [ ] Write a report that separately adjudicates the exact capacity gate, learned structural gate, and causal gate without overstating autonomous routing or general intelligence.
- [ ] Commit verified consolidated results and rerun final CI.
