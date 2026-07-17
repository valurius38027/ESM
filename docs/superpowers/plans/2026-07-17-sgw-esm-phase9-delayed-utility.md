# SGW-ESM Phase 9 Delayed Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and formally validate a variable-delay scarce-workspace task that measures long-horizon retention credit and compares direct versus curriculum annealing.

**Architecture:** Extend the Phase 8 retention dataset with an explicit delay block of context-irrelevant binding pairs and metadata, while keeping one retention forward path that parses variable pair counts. Add Phase 9 presets, delay-aware training streams, evaluation telemetry, causal interventions, strict aggregation, and a six-shard Actions matrix.

**Tech Stack:** C++20, CMake, Ninja, Python standard library, Bash, GitHub Actions, CPU only.

## Global Constraints

- Preserve all Phase 1-8 condition names, outputs, parameter counts, and tests.
- Runtime remains standard-library-only and CPU-only.
- Formal inference is hard top-1; final 450 updates have zero soft/hard disagreement.
- Formal sample stream is deterministic and non-repeating for 14,400 samples per learned condition.
- Workflow is manual after the first formal run.

---

### Task 1: Delay-aware dataset

**Files:**
- Modify: `include/sgw/dataset.hpp`
- Modify: `src/dataset.cpp`
- Modify: `tests/test_dataset.cpp`

**Interfaces:**
- Produces `DelayedRetentionTaskConfig`, `DelayedRetentionDataset`, and `DelayedRetentionBindingStream`.
- `BindingSample` gains `delay_binding_count`, `source_query_distance`, and `binding_is_delay_distractor`.

- [ ] Add failing tests that require deterministic unique samples with six candidates plus configurable distractor pairs, all distractors context-irrelevant, and correct source-query distance.
- [ ] Run `cmake --build build-red --target sgw_tests && ./build-red/sgw_tests --filter delayed_retention` and confirm missing-symbol failures.
- [ ] Implement configuration validation, vocabulary reuse, split generation, and exact-budget stream generation.
- [ ] Re-run the focused tests and the full `sgw_tests` binary.
- [ ] Commit `feat: add delayed retention task distribution`.

### Task 2: Variable-length retention forward path

**Files:**
- Modify: `include/sgw/config.hpp`
- Modify: `include/sgw/model.hpp`
- Modify: `src/config.cpp`
- Modify: `src/model.cpp`
- Modify: `tests/test_model_forward.cpp`
- Modify: `tests/test_interventions.cpp`

**Interfaces:**
- `ModelConfig` gains `retention_candidate_binding_count`.
- Retention forward accepts `context + N pairs + query`, where `N >= retention_candidate_binding_count`.
- `StepTrace` records distractor decisions and relevant survival after the delay block.

- [ ] Add failing forward tests for delay lengths 0, 6, 18, and 24 using one preset.
- [ ] Verify failures arise from the Phase 8 fixed-length check.
- [ ] Implement dynamic pair parsing without a model-visible distractor flag.
- [ ] Add interventions for relevant-looking distractors and reversed delay order.
- [ ] Run focused and full C++ tests.
- [ ] Commit `feat: support delayed retention sequences`.

### Task 3: Phase 9 presets and curriculum stream

**Files:**
- Modify: `include/sgw/presets.hpp`
- Modify: `src/presets.cpp`
- Modify: `include/sgw/experiment.hpp`
- Modify: `src/experiment.cpp`
- Modify: `tests/test_presets.cpp`
- Modify: `tests/test_training.cpp`
- Modify: `tests/test_gradients.cpp`

**Interfaces:**
- Adds six `kv_delayed_*` conditions and presets.
- Adds `DelayedRetentionBindingStream::set_delay_binding_count` or an equivalent deterministic curriculum API.
- `TrainingHistory` records delay and delayed-retention telemetry per update.

- [ ] Add failing tests for names, parameter parity, direct delay, the 15%/15%/20%/50% delay curriculum, exact sample budget, and final hard-only gradients.
- [ ] Verify red failures.
- [ ] Implement presets and curriculum training while keeping direct and curriculum annealed parameter-identical.
- [ ] Run focused and full C++ tests.
- [ ] Commit `feat: add phase 9 delayed retention policies`.

### Task 4: CLI and multi-delay evaluation

**Files:**
- Modify: `apps/sgw_train.cpp`
- Modify: `tests/test_cli.sh`

**Interfaces:**
- CLI accepts all Phase 9 conditions.
- Primary CSV retains existing columns and appends delayed metrics plus evaluation summaries for delays 0, 6, 12, 18, and 24.
- Causal CSV includes the Phase 9 interventions.

- [ ] Add CLI assertions for deterministic primary and causal CSV output.
- [ ] Confirm the CLI test fails before implementation.
- [ ] Implement Phase 9 task dispatch, curriculum stream construction, five-delay evaluation, and CSV serialization.
- [ ] Run `tests/test_cli.sh build-red /tmp/phase9-cli` twice and compare outputs.
- [ ] Commit `feat: expose phase 9 delayed utility experiments`.

### Task 5: Strict aggregation and formal workflow

**Files:**
- Create: `scripts/aggregate_phase9.py`
- Create: `scripts/run_phase9_shard.sh`
- Create: `tests/test_phase9_summary.py`
- Modify: `CMakeLists.txt`
- Create: `.github/workflows/phase9-experiments.yml`

**Interfaces:**
- Aggregator consumes exactly 180 primary files and 30 causal files for 30 seeds and six conditions.
- Produces runs, causal, summary, paired, causal comparisons, delay curves, retention, manifest, and SHA256SUMS.

- [ ] Write a synthetic 30-seed regression that enforces every preregistered gate and rejects missing, extra, duplicate, malformed, or wrong-budget input.
- [ ] Verify the test fails because the aggregator does not exist.
- [ ] Implement deterministic 50,000-replicate paired bootstrap and strict file-set validation.
- [ ] Add the six-shard 30-seed × 1,800-step workflow.
- [ ] Run all summary tests and a two-seed smoke matrix.
- [ ] Commit `test: add phase 9 formal experiment matrix`.

### Task 6: Verification and formal evidence

**Files:**
- Create after execution: `results/PHASE9_VALIDATION.md`
- Create after execution: `results/phase9-formal/*`

**Interfaces:**
- Uses the workflow artifact as the authoritative remote result.
- Independent verification uses 200,000 bootstrap replicates with a different RNG.

- [ ] Run GCC Debug, Clang Debug, GCC Release, and ASan/UBSan gates.
- [ ] Run the six-shard formal workflow.
- [ ] Verify artifact outer digest and internal SHA256SUMS.
- [ ] Re-run the repository aggregator and require byte-identical consolidated files.
- [ ] Independently recompute point estimates and bootstrap interval signs.
- [ ] Write the scientific report with explicit pass/fail status for capability, robustness, extrapolation, curriculum, and causal gates.
- [ ] Commit results and keep the PR draft until the next phase decision.
