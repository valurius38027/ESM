# SGW-ESM Phase 9 Stability and Delayed Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Phase 8 annealed retention capability into a seed-robust policy, then test that stabilized discrete policy under variable and substantially longer utility delays.

**Architecture:** Phase 9 is deliberately divided into two sequential gates. Phase 9A keeps the Phase 8 task unchanged and modifies only retention-router optimization, so convergence failures can be attributed. Phase 9B freezes the winning stabilization recipe and changes only the task horizon by adding variable binding counts and query delays. Both gates retain tied entity/value identities, three workspace slots, hard forward actions, and a final hard-only window.

**Tech Stack:** C++20, CMake 3.20+, Ninja, Python 3 standard library, Bash, GitHub Actions on CPU-only Ubuntu runners.

## Global Constraints

- Branch from `agent/phase8-retention-eviction`; do not rewrite or merge the Phase 8 draft PR.
- Preserve all Phase 1–8 conditions, tests, reports, and formal result files.
- Runtime remains standard-library-only and CPU-only.
- Every condition must be deterministic for a fixed seed.
- CSV and JSON output must use LF line endings and deterministic field ordering.
- Formal aggregation must reject missing, duplicate, stale, and extra files.
- Every formal result directory must include and pass `SHA256SUMS`.
- Causal interventions remain evaluation-only.
- The final 300 updates and inference must use hard one-hot retention actions with zero soft/hard disagreement.
- Do not introduce delayed utility until Phase 9A selects and validates a stabilization recipe.
- Keep Phase 9A and Phase 9B in separate commits, workflows, result directories, and scientific reports.

---

## File map

### New files

- `docs/superpowers/specs/2026-07-17-sgw-esm-phase9-stability-design.md` — preregistered Phase 9A design.
- `docs/superpowers/specs/2026-07-17-sgw-esm-phase9-delayed-utility-design.md` — preregistered Phase 9B design.
- `scripts/run_phase9a_shard.sh` — paired Phase 9A stability conditions.
- `scripts/aggregate_phase9a.py` — stability, tail-risk, trajectory, and causal aggregation.
- `tests/test_phase9a_summary.py` — deterministic synthetic aggregation regression.
- `.github/workflows/phase9a-stability.yml` — manual formal stability matrix.
- `scripts/run_phase9b_shard.sh` — delayed-utility conditions and delay buckets.
- `scripts/aggregate_phase9b.py` — per-delay and overall paired aggregation.
- `tests/test_phase9b_summary.py` — delayed-utility aggregation regression.
- `.github/workflows/phase9b-delayed-utility.yml` — manual delayed-utility matrix.
- `results/PHASE9A_VALIDATION.md` and `results/PHASE9B_VALIDATION.md` — bounded reports.

### Modified files

- `include/sgw/config.hpp`, `src/config.cpp` — stabilization and delayed-task invariants.
- `include/sgw/presets.hpp`, `src/presets.cpp` — named conditions and presets.
- `include/sgw/dataset.hpp`, `src/dataset.cpp` — variable-delay retention samples and streams.
- `include/sgw/model.hpp`, `src/model.cpp` — normalized initialization, entropy surrogate, router gradient scaling, and trajectory trace fields.
- `include/sgw/experiment.hpp`, `src/experiment.cpp` — router-group gradient processing, checkpoint metrics, and delayed streams.
- `apps/sgw_train.cpp` — new conditions and optional trajectory CSV.
- `tests/test_config_optimizer.cpp`, `test_presets.cpp`, `test_dataset.cpp`, `test_model_forward.cpp`, `test_gradients.cpp`, `test_training.cpp`, `test_interventions.cpp`, `test_cli.sh`.
- `CMakeLists.txt`, `.gitignore`, `README.md`.

---

# Phase 9A — Seed-Robust Retention Optimization

## Preregistered conditions

The Phase 8 task, data split, 1200-update budget, batch size 8, and three-slot workspace remain unchanged.

- `p9a_phase8_reference`: exact Phase 8 annealed retention behavior.
- `p9a_equal_norm_init`: context-codebook Xavier rows normalized to identical L2 norm after initialization.
- `p9a_entropy_floor`: reference initialization plus an early normalized-entropy floor.
- `p9a_router_grad_scale`: reference initialization with router-group gradients multiplied by `0.25` before global clipping.
- `p9a_combined`: equal-norm initialization, entropy floor, and router gradient scale together. This is the preregistered primary candidate; component conditions are diagnostic.

Entropy-floor schedule:

- updates `0..599`: normalized action-entropy target `0.65`;
- updates `600..899`: target decreases linearly from `0.65` to `0`;
- updates `900..1199`: no entropy term and no soft surrogate.

The normalized entropy is `H(p) / log(number_of_available_actions)`. The auxiliary loss is:

```cpp
const double deficit = std::max(0.0, target - normalized_entropy.value());
loss = loss + tape.constant(weight) * tape.constant(deficit) *
              tape.constant(deficit);
```

The numeric deficit above is intentionally stop-gradient. The actual implementation must use a differentiable hinge:

```cpp
const ad::Var deficit = ad::maximum(
    tape.constant(0.0), tape.constant(target) - normalized_entropy);
loss = loss + tape.constant(weight) * deficit * deficit;
```

If `ad::maximum` does not exist, add a tested differentiable `relu` helper to `autodiff.hpp/.cpp` rather than silently using a numeric constant.

Formal Phase 9A primary matrix:

- 60 paired seeds;
- `p9a_phase8_reference` and `p9a_combined` are primary;
- all five conditions run for diagnostic completeness;
- 1200 updates, batch 8, 160 train diagnostics, 160 holdout diagnostics;
- 10 GitHub Actions shards of 6 seeds;
- 50,000 bootstrap replicates.

Primary success gates for `p9a_combined`:

1. mean holdout accuracy at least `0.97`;
2. median holdout accuracy at least `0.99`;
3. minimum holdout accuracy at least `0.85`;
4. at least 57/60 seeds reach `0.90`;
5. at least 45/60 seeds reach `0.99`;
6. mean query-hit rate at least `0.96`;
7. mean relevant-eviction rate below `0.02`;
8. worst-decile accuracy significantly exceeds the Phase 8 reference;
9. no regression in mean NLL or Brier versus the reference;
10. final 300-update soft/hard disagreement is exactly zero for every seed.

---

### Task 1: Preregister Phase 9A and add configuration

**Files:**
- Create: `docs/superpowers/specs/2026-07-17-sgw-esm-phase9-stability-design.md`
- Modify: `include/sgw/config.hpp`
- Modify: `src/config.cpp`
- Test: `tests/test_config_optimizer.cpp`

**Interfaces:**
- Produces: `RetentionStabilizationMode`, `ModelConfig::retention_entropy_target`, `retention_entropy_weight`, `retention_entropy_decay_start`, `retention_router_gradient_scale`, and `retention_equal_norm_initialization`.

- [ ] **Step 1: Write failing configuration tests**

Add tests equivalent to:

```cpp
SGW_TEST(phase9a_stabilization_config_rejects_invalid_values) {
  auto config = sgw::make_model_config(
      sgw::ModelPreset::p9a_combined, 20, 6);
  config.retention_entropy_target = 1.1;
  SGW_REQUIRE_THROWS(config.validate());
  config = sgw::make_model_config(sgw::ModelPreset::p9a_combined, 20, 6);
  config.retention_router_gradient_scale = 0.0;
  SGW_REQUIRE_THROWS(config.validate());
}
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
cmake -S . -B build-phase9a-red -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-phase9a-red --parallel 2
./build-phase9a-red/sgw_tests --filter phase9a
```

Expected: compilation failure because the new configuration fields and preset do not exist.

- [ ] **Step 3: Add the exact configuration interface**

```cpp
enum class RetentionStabilizationMode {
  none,
  equal_norm_initialization,
  entropy_floor,
  router_gradient_scale,
  combined,
};

struct ModelConfig {
  // Existing fields remain unchanged.
  RetentionStabilizationMode retention_stabilization{
      RetentionStabilizationMode::none};
  double retention_entropy_target{0.0};
  double retention_entropy_weight{0.0};
  std::size_t retention_entropy_decay_start{600};
  double retention_router_gradient_scale{1.0};
  bool retention_equal_norm_initialization{false};
};
```

Validation must enforce:

- entropy target in `[0, 1]`;
- entropy weight finite and non-negative;
- decay start no greater than router anneal steps;
- gradient scale finite and strictly positive;
- stabilization modes only valid for learned retention.

- [ ] **Step 4: Run focused and full C++ tests**

```bash
cmake --build build-phase9a-red --parallel 2
./build-phase9a-red/sgw_tests --filter phase9a
./build-phase9a-red/sgw_tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-17-sgw-esm-phase9-stability-design.md \
  include/sgw/config.hpp src/config.cpp tests/test_config_optimizer.cpp
git commit -m "feat: add Phase 9A stabilization configuration"
```

---

### Task 2: Add Phase 9A presets and equal-norm initialization

**Files:**
- Modify: `include/sgw/presets.hpp`
- Modify: `src/presets.cpp`
- Modify: `src/model.cpp`
- Test: `tests/test_presets.cpp`
- Test: `tests/test_model_forward.cpp`

**Interfaces:**
- Produces conditions `p9a_phase8_reference`, `p9a_equal_norm_init`, `p9a_entropy_floor`, `p9a_router_grad_scale`, and `p9a_combined`.
- Produces helper `normalize_parameter_rows(Parameter&)` local to `model.cpp`.

- [ ] **Step 1: Write failing preset and initialization tests**

Verify all conditions parse, use three slots, use annealed learned retention, and retain the Phase 8 parameter count. Add a test that constructs `p9a_equal_norm_init` and asserts all `kv_context_codebook` row norms are equal within `1e-12`.

- [ ] **Step 2: Run tests and verify RED**

```bash
cmake --build build-phase9a-red --parallel 2
./build-phase9a-red/sgw_tests --filter phase9a
```

- [ ] **Step 3: Implement presets**

`p9a_phase8_reference` must call the existing Phase 8 annealed preset without changing any field. The other presets change only stabilization fields. Do not change key/value dimensions, workspace slots, retention mode, logit scale, or temperature schedule.

After Xavier initialization of `kv_context_codebook_`, equal-norm initialization must execute:

```cpp
for (std::size_t row = 0; row < parameter.rows(); ++row) {
  double squared_norm = 0.0;
  for (std::size_t column = 0; column < parameter.columns(); ++column) {
    const double value = parameter.value(row * parameter.columns() + column);
    squared_norm += value * value;
  }
  const double scale = 1.0 / std::sqrt(std::max(squared_norm, 1.0e-12));
  for (std::size_t column = 0; column < parameter.columns(); ++column) {
    parameter.value(row * parameter.columns() + column) *= scale;
  }
}
```

- [ ] **Step 4: Verify byte-stable construction and full tests**

Construct the same model twice with the same seed and compare all parameter bytes. Run all C++ tests.

- [ ] **Step 5: Commit**

```bash
git add include/sgw/presets.hpp src/presets.cpp src/model.cpp \
  tests/test_presets.cpp tests/test_model_forward.cpp
git commit -m "feat: add Phase 9A stabilization presets"
```

---

### Task 3: Add differentiable entropy-floor loss and router gradient scaling

**Files:**
- Modify: `include/sgw/autodiff.hpp`
- Modify: `src/autodiff.cpp`
- Modify: `include/sgw/model.hpp`
- Modify: `src/model.cpp`
- Modify: `include/sgw/experiment.hpp`
- Modify: `src/experiment.cpp`
- Test: `tests/test_autodiff.cpp`
- Test: `tests/test_gradients.cpp`
- Test: `tests/test_training.cpp`

**Interfaces:**
- Produces `ad::relu(Var)` if absent.
- Produces `SequenceResult::retention_entropy_auxiliary`.
- Produces `TrainingHistory::retention_entropy_loss`.
- Produces helper `scale_retention_router_gradients(SgwEsmModel&, double)`.

- [ ] **Step 1: Add RED tests**

Tests must prove:

1. `relu` forward and gradient behavior at negative and positive inputs;
2. entropy-floor loss yields nonzero context-codebook gradients before update 900;
3. the auxiliary loss is exactly zero at and after update 900;
4. router-gradient scaling affects only `kv_context_codebook` and `kv_retention_age_weight` gradients;
5. the Phase 8 reference produces byte-identical history to the pre-Phase 9 implementation.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cmake --build build-phase9a-red --parallel 2
./build-phase9a-red/sgw_tests --filter entropy
./build-phase9a-red/sgw_tests --filter gradient
```

- [ ] **Step 3: Implement the entropy schedule**

Add a pure helper:

```cpp
double retention_entropy_target_at_step(
    const ModelConfig& config, std::size_t step) noexcept;
```

It returns the configured target through step 599, linearly decays over steps 600–899, and returns zero from step 900 onward.

The forward path must retain an `ad::Var` normalized entropy for each retention decision. Average the hinge-squared penalty over decisions. Add it to total training loss only when the target and weight are positive.

- [ ] **Step 4: Implement router-group gradient scaling**

After batch gradient averaging and before global norm measurement/clipping:

```cpp
for (Parameter* parameter : model.parameters().parameters()) {
  if (parameter->name() == "kv_context_codebook" ||
      parameter->name() == "kv_retention_age_weight") {
    for (std::size_t index = 0; index < parameter->size(); ++index) {
      parameter->gradient(index) *= config.retention_router_gradient_scale;
    }
  }
}
```

Do not scale entity or value codebooks.

- [ ] **Step 5: Run full tests and commit**

```bash
./build-phase9a-red/sgw_tests

git add include/sgw/autodiff.hpp src/autodiff.cpp include/sgw/model.hpp \
  src/model.cpp include/sgw/experiment.hpp src/experiment.cpp \
  tests/test_autodiff.cpp tests/test_gradients.cpp tests/test_training.cpp
git commit -m "feat: stabilize Phase 9A retention optimization"
```

---

### Task 4: Add checkpoint trajectories and tail-risk metrics

**Files:**
- Modify: `include/sgw/experiment.hpp`
- Modify: `src/experiment.cpp`
- Modify: `apps/sgw_train.cpp`
- Modify: `tests/test_cli.sh`
- Test: `tests/test_training.cpp`

**Interfaces:**
- Produces `TrainingCheckpoint` and `TrainingHistory::checkpoints`.
- Adds CLI option `--trajectory-output PATH`.
- Emits one checkpoint every 50 updates plus the final update.

- [ ] **Step 1: Write RED tests**

A 120-update run must emit checkpoints at updates `49`, `99`, and `119`. Each row must contain:

```text
condition,seed,update,primary_loss,entropy_loss,gradient_norm,
query_hit_rate,relevant_eviction_rate,write_rate,skip_rate,
action_entropy,soft_hard_disagreement
```

Run the same command twice and compare trajectory files byte-for-byte.

- [ ] **Step 2: Implement checkpoint collection**

Evaluate a fixed 64-sample diagnostic training set and 64-sample holdout set at each checkpoint. Do not consume the online stream for checkpoint evaluation.

- [ ] **Step 3: Add tail metrics to aggregation contracts**

The primary run CSV must include:

- first update where normalized action entropy falls below `0.10` for three consecutive checkpoints;
- final 300-update query-hit rate;
- final 300-update relevant-eviction rate;
- minimum checkpoint holdout accuracy;
- final holdout accuracy.

- [ ] **Step 4: Run CLI and full tests, then commit**

```bash
bash tests/test_cli.sh build-phase9a-red/sgw_train build-phase9a-red
./build-phase9a-red/sgw_tests

git add include/sgw/experiment.hpp src/experiment.cpp apps/sgw_train.cpp \
  tests/test_cli.sh tests/test_training.cpp
git commit -m "feat: add Phase 9A convergence trajectories"
```

---

### Task 5: Add Phase 9A formal aggregation and workflow

**Files:**
- Create: `scripts/run_phase9a_shard.sh`
- Create: `scripts/aggregate_phase9a.py`
- Create: `tests/test_phase9a_summary.py`
- Create: `.github/workflows/phase9a-stability.yml`
- Modify: `CMakeLists.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces `phase9a_runs.csv`, `phase9a_trajectories.csv`, `phase9a_summary.csv`, `phase9a_paired.csv`, `phase9a_tail.csv`, `phase9a_causal_comparisons.csv`, `phase9a_manifest.json`, and `SHA256SUMS`.

- [ ] **Step 1: Write a synthetic 60-seed aggregation regression**

The fixture must include all five conditions and demonstrate a combined recipe that improves the worst decile while preserving mean NLL/Brier. The test must reject one extra seed file, one duplicate condition, and one missing trajectory checkpoint.

- [ ] **Step 2: Implement tail-risk aggregation**

Compute:

- mean, standard deviation, median, minimum, 10th percentile;
- count at accuracy thresholds 0.90 and 0.99;
- worst-decile mean accuracy;
- paired bootstrap for mean metrics;
- paired bootstrap for the worst-decile difference by resampling paired seeds and recomputing both worst deciles.

- [ ] **Step 3: Add a manual formal workflow**

Use 10 shards, each with 6 seeds, `timeout-minutes: 240`, Release builds, and 50,000 bootstrap replicates. The workflow must use only `workflow_dispatch` after its first validation run.

- [ ] **Step 4: Run a 4-seed smoke matrix**

```bash
scripts/run_phase9a_shard.sh build-release 0 4 1200 8 160 160 \
  results/phase9a-smoke
python3 scripts/aggregate_phase9a.py \
  --raw-dir results/phase9a-smoke/raw \
  --trajectory-dir results/phase9a-smoke/trajectories \
  --causal-dir results/phase9a-smoke/causal \
  --output-dir results/phase9a-smoke/formal \
  --seed-start 0 --seed-count 4 --bootstrap-replicates 2000
(cd results/phase9a-smoke/formal && sha256sum --check SHA256SUMS)
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_phase9a_shard.sh scripts/aggregate_phase9a.py \
  tests/test_phase9a_summary.py .github/workflows/phase9a-stability.yml \
  CMakeLists.txt .gitignore
git commit -m "ci: add Phase 9A seed stability matrix"
```

---

### Task 6: Execute Phase 9A and freeze the recipe

**Files:**
- Create: `results/PHASE9A_VALIDATION.md`
- Create: `results/phase9a-formal/*`
- Modify: `README.md`

- [ ] **Step 1: Run four engineering gates**

Run GCC Debug, Clang Debug, GCC Release, and ASan/UBSan. Split commands to respect host command timeouts.

- [ ] **Step 2: Push a draft Phase 9 branch and run the 60-seed matrix**

Use a new branch `agent/phase9-stability-delayed-utility`. Do not alter PR #6.

- [ ] **Step 3: Download and independently reaggregate**

Verify artifact ZIP SHA-256, internal `SHA256SUMS`, raw row counts, trajectory counts, and byte-identical regenerated formal files. Use a separate RNG for a 200,000-replicate independent bootstrap.

- [ ] **Step 4: Apply the preregistered recipe decision**

Proceed to Phase 9B only if `p9a_combined` passes every primary gate. If it fails, stop and write a negative report; do not select the best component post hoc as the official recipe.

- [ ] **Step 5: Commit the report and results**

```bash
git add results/PHASE9A_VALIDATION.md README.md
git add -f results/phase9a-formal
git commit -m "results: record Phase 9A stability validation"
```

---

# Phase 9B — Variable Delayed Utility

Phase 9B may begin only after Phase 9A passes. The `p9a_combined` configuration becomes the frozen `p9b_stable_annealed` recipe.

## Delayed task

Each episode contains:

- one context token;
- a variable binding count sampled deterministically from `{8, 10, 12}`;
- exactly three context-relevant entities and the remaining bindings irrelevant;
- three workspace slots;
- a delay bucket sampled from `{0, 8, 32, 64}`;
- delay tokens after the last binding and before the query marker;
- one final query sampled from the relevant entities.

Delay tokens must be semantically inert and use a dedicated filler role. They must not alter context, keys, values, or workspace. The task must preserve entity-value structural holdout and no-replacement online training.

Phase 9B conditions:

- `p9b_full_capacity`
- `p9b_oracle`
- `p9b_fifo`
- `p9b_reservoir`
- `p9b_hard_retention`
- `p9b_stable_annealed`

Formal protocol:

- 30 paired seeds;
- 1600 updates, batch 8;
- final 300 updates hard-only;
- balanced diagnostic holdout with 64 samples per delay bucket per seed;
- six 5-seed Actions shards;
- 50,000 bootstrap replicates.

Primary gates:

1. oracle accuracy at least 0.99 in every delay bucket;
2. stable annealed overall accuracy at least 0.90;
3. stable annealed accuracy at delay 64 at least 0.85;
4. stable annealed significantly exceeds FIFO and reservoir in NLL, Brier, and accuracy overall and at delay 64;
5. query-hit rate at least 0.90 overall and at least 0.85 at delay 64;
6. relevant-eviction rate below 0.05 overall;
7. all causal interventions are harmful;
8. final 300-update disagreement is zero for every seed.

---

### Task 7: Implement the delayed-retention dataset

**Files:**
- Create: `docs/superpowers/specs/2026-07-17-sgw-esm-phase9-delayed-utility-design.md`
- Modify: `include/sgw/dataset.hpp`
- Modify: `src/dataset.cpp`
- Test: `tests/test_dataset.cpp`

**Interfaces:**
- Produces `DelayedRetentionTaskConfig`, `DelayedRetentionDataset`, `DelayedRetentionStream`, and `DelayBucket`.

- [ ] **Step 1: Add RED tests**

For each delay bucket, verify exact sequence length, exactly three relevant bindings, correct query target, structural holdout, and no repeated online samples across 12,800 draws.

- [ ] **Step 2: Add the exact task interface**

```cpp
enum class DelayBucket : std::size_t { zero = 0, short_8, medium_32, long_64 };

struct DelayedRetentionTaskConfig {
  std::size_t context_count{3};
  std::size_t entity_count{18};
  std::size_t value_count{6};
  std::size_t relevant_binding_count{3};
  std::size_t workspace_slots{3};
  std::array<std::size_t, 3> binding_counts{8, 10, 12};
  std::array<std::size_t, 4> delays{0, 8, 32, 64};

  void validate() const;
};
```

`BindingSample` must record `delay_bucket`, `delay_length`, and `binding_count` for aggregation.

- [ ] **Step 3: Implement deterministic generation**

Use the existing bounded RNG and literal-key uniqueness machinery. Delay filler tokens must be appended between the final value and query marker.

- [ ] **Step 4: Run focused/full tests and commit**

```bash
./build-phase9a-red/sgw_tests --filter delayed
./build-phase9a-red/sgw_tests

git add docs/superpowers/specs/2026-07-17-sgw-esm-phase9-delayed-utility-design.md \
  include/sgw/dataset.hpp src/dataset.cpp tests/test_dataset.cpp
git commit -m "feat: add Phase 9B delayed retention task"
```

---

### Task 8: Integrate delayed evaluation, formal matrix, and final report

**Files:**
- Modify: `include/sgw/presets.hpp`, `src/presets.cpp`
- Modify: `include/sgw/experiment.hpp`, `src/experiment.cpp`
- Modify: `apps/sgw_train.cpp`, `tests/test_cli.sh`
- Create: `scripts/run_phase9b_shard.sh`
- Create: `scripts/aggregate_phase9b.py`
- Create: `tests/test_phase9b_summary.py`
- Create: `.github/workflows/phase9b-delayed-utility.yml`
- Modify: `CMakeLists.txt`, `.gitignore`, `README.md`
- Create: `results/PHASE9B_VALIDATION.md`, `results/phase9b-formal/*`

**Interfaces:**
- Primary CSV adds `binding_count`, `delay_bucket`, and per-delay metric summaries.
- Aggregator emits `phase9b_delay_summary.csv` and `phase9b_delay_paired.csv` in addition to normal formal files.

- [ ] **Step 1: Write RED CLI and aggregation tests**

Require one row for each condition × seed and one balanced per-delay diagnostic table. Reject files where a delay bucket has fewer than 64 holdout samples.

- [ ] **Step 2: Add Phase 9B presets and stream training**

`p9b_stable_annealed` must exactly reuse the frozen Phase 9A combined optimization fields. Do not retune them after viewing delayed-task results.

- [ ] **Step 3: Add per-delay evaluation**

Evaluate and persist NLL, accuracy, Brier, ECE, query-hit rate, relevant-eviction rate, and mean retained age separately for delay 0, 8, 32, and 64.

- [ ] **Step 4: Add formal aggregation and manual workflow**

Compute overall and per-delay paired intervals. Use six 5-seed shards and 50,000 replicates. Upload raw, causal, trajectory, and formal directories.

- [ ] **Step 5: Run smoke, engineering gates, and the 30-seed matrix**

Do not interpret smoke results scientifically. Download the final artifact and independently reaggregate all files.

- [ ] **Step 6: Commit a bounded report**

The report must distinguish:

- seed robustness on the unchanged Phase 8 task;
- degradation as a function of delay;
- causal usefulness of retention;
- whether hard-from-start remains inferior after stabilization;
- any failure bucket or seed tail.

```bash
git add results/PHASE9B_VALIDATION.md README.md
git add -f results/phase9b-formal
git commit -m "results: record Phase 9B delayed utility validation"
```

---

## Final verification checklist

Before claiming Phase 9 complete:

```bash
git diff --check
cmake -S . -B build-gcc -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-gcc --parallel 2
ctest --test-dir build-gcc --output-on-failure

# Repeat for Clang, Release, and ASan/UBSan.

(cd results/phase9a-formal && sha256sum --check SHA256SUMS)
(cd results/phase9b-formal && sha256sum --check SHA256SUMS)
git status --short
```

The final branch must remain a draft research PR until the user explicitly requests integration. Preserve failed conditions and negative reports in the repository.
