# ESM Engineering Handoff — Phase 8 Checkpoint

**Handoff date:** 2026-07-17  
**Repository:** `https://github.com/valurius38027/ESM`  
**Canonical branch:** `agent/phase8-retention-eviction`  
**Draft PR:** `https://github.com/valurius38027/ESM/pull/6`  
**PR base:** `agent/phase7-learned-slot-routing`  
**Validated Phase 8 head before handoff documents:** `444363a4b154037b80b0a33d9197705666a9639d`

This document is the primary context handoff for a fresh agent or engineer. The transient container workspace used during implementation was cleaned by the host; do not rely on historical `/mnt/data/...` paths mentioned in chat. Restore from the handoff Git bundle or clone the canonical remote branch.

---

## 1. Current project state

Phase 8 is implemented, formally evaluated, independently reaggregated, committed to the remote branch, and covered by final four-configuration CI. The branch remains a draft PR intentionally: it is a research checkpoint, not a production merge candidate.

The central result is that a sparse three-slot workspace can learn context-dependent retention and eviction under genuine capacity scarcity. Straight-through soft-to-hard annealing strongly outperforms FIFO, seeded reservoir sampling, and the identical policy trained hard from the first update. The final 300 training updates and inference are fully discrete.

The main unresolved issue is convergence robustness across seeds. Mean accuracy is high, but a minority of seeds converge to materially worse retention policies.

### Canonical remote facts

- Branch: `agent/phase8-retention-eviction`
- Draft PR: `#6`
- Formal workflow run: `29559003157`
- Formal artifact ID: `8398546609`
- Formal artifact ZIP SHA-256: `3bb581b4b104f29e836d4cdbce0cb519727466c8df17e408c3595e1e5744300c`
- Formal source head: `0daf86d4717fd2f800aef1b58443efb228cbf51b`
- Final CI run after result materialization: `29559975319`
- Final CI status: GCC Debug, Clang Debug, GCC Release, and ASan/UBSan all passed
- Focused C++ tests: 85

The handoff document commits advance the branch beyond the validated Phase 8 head. Use `git log -1` after restoration to identify the final handoff commit.

---

## 2. How to restore the workspace

### Preferred: restore from the handoff bundle

```bash
mkdir ESM
cd ESM
git clone /path/to/ESM_phase8_handoff.git.bundle .
git switch agent/phase8-retention-eviction
```

If the clone chooses another branch, list refs and switch explicitly:

```bash
git branch -a
git switch agent/phase8-retention-eviction
```

Verify the bundle before cloning when needed:

```bash
git bundle verify /path/to/ESM_phase8_handoff.git.bundle
```

### Alternative: clone the remote branch

```bash
git clone --branch agent/phase8-retention-eviction \
  --single-branch https://github.com/valurius38027/ESM.git ESM
cd ESM
```

Some OpenAI host containers intermittently cannot resolve `github.com`. In that environment, use the bundle or create a temporary GitHub Actions export workflow rather than repeatedly retrying network clones.

### Establish a new research worktree

Do not modify the Phase 8 branch directly for Phase 9 implementation:

```bash
git switch agent/phase8-retention-eviction
git worktree add .worktrees/phase9-delayed-utility \
  -b feature/phase9-delayed-utility
cd .worktrees/phase9-delayed-utility
```

Ensure `.worktrees/` is ignored before creating it. Do not merge PR #6 automatically.

---

## 3. Repository architecture

The project is a C++20, CPU-only research prototype with no runtime dependency beyond the standard library.

### Build and entry points

- `CMakeLists.txt` — library, executable, and CTest registration.
- `apps/sgw_train.cpp` — deterministic experiment CLI and CSV emission.
- `cmake/Warnings.cmake` — compiler warning policy.

### Public model and experiment interfaces

- `include/sgw/config.hpp` — model invariants, routing modes, retention modes, and validation.
- `include/sgw/dataset.hpp` — binding tasks, structural holdout streams, and Phase 8 scarce-retention episodes.
- `include/sgw/model.hpp` — model state, forward interventions, route traces, KV/retention forward paths.
- `include/sgw/experiment.hpp` — training/evaluation metrics, online stream overloads, and histories.
- `include/sgw/presets.hpp` — model presets and named experiment conditions.
- `include/sgw/autodiff.hpp`, `optimizer.hpp`, `parameter.hpp`, `tensor.hpp`, `topk.hpp` — self-contained scalar autodiff and optimization substrate.

### Implementations

- `src/config.cpp`
- `src/dataset.cpp`
- `src/model.cpp`
- `src/experiment.cpp`
- `src/presets.cpp`
- supporting autodiff, tensor, optimizer, parameter, and top-k sources.

`src/model.cpp` is intentionally large because earlier phases preserved one auditable model implementation. Do not perform an unrelated refactor while adding Phase 9; split only when a new responsibility has a clear test boundary.

### Formal experiment tools

- `scripts/run_phase8_shard.sh` — runs six paired Phase 8 conditions for a seed range.
- `scripts/aggregate_phase8.py` — strict file-set validation, paired bootstrap, causal gates, manifests, and checksums.
- `.github/workflows/phase8-experiments.yml` — manual six-shard formal matrix.
- `.github/workflows/ci.yml` — four-configuration engineering validation.

Earlier phase scripts and aggregators remain in the repository as reproducibility records. Do not delete them merely because Phase 8 is current.

### Tests

- `tests/test_dataset.cpp` — task generation, structural isolation, and stream uniqueness.
- `tests/test_model_forward.cpp` — exact route/state behavior.
- `tests/test_interventions.cpp` — causal path changes.
- `tests/test_gradients.cpp` — finite differences and router-gradient boundaries.
- `tests/test_training.cpp` — deterministic optimization and telemetry accounting.
- `tests/test_cli.sh` — byte-stable CLI and CSV contracts.
- `tests/test_phase*_summary.py` — deterministic aggregation and stale-file rejection.

### Formal evidence

- `results/PHASE8_VALIDATION.md` — bounded scientific report.
- `results/phase8-formal/` — consolidated formal output, independent verification, and `SHA256SUMS`.
- Earlier `results/PHASE*_VALIDATION.md` files preserve the negative and positive experimental history.

---

## 4. Phase-by-phase scientific history

Do not skip this history: several apparently promising designs were falsified, and repeating them would waste substantial time.

### Phase 1 — minimal sparse global workspace

Implemented a recurrent spine, persistent mechanisms, hard top-k activation, sparse workspace writes, and sparse broadcast. The system trained end-to-end and was deterministic, but did not outperform a recurrent baseline under the initial comparison.

### Phase 2 — fair baselines and causal controls

Added parameter-matched and compute-matched recurrent controls plus evaluation-only interventions. SGW did not pass the superiority gate. Removing broadcast slightly improved NLL, showing that the broadcast path was not productively used.

### Phase 3 — auxiliary credit assignment

Added forced broadcast topology and temporary auxiliary supervision. The intervention path still failed to become reliably beneficial. Free-form mechanism output often increased overconfidence.

### Phase 4 — information-separated fixed mediation

Removed the recurrent content bypass and forced information through a fixed sparse workspace. Causal dependence appeared, but probability modeling remained poor and overconfident. Capacity existed; representation and identity preservation were inadequate.

### Phase 5 — structural holdout and calibration

Added deterministic no-replacement online training, entity-value structural holdout, Brier/ECE metrics, and bounded logits. Bounding logits improved calibration but did not solve structural generalization. Free-form mediation learned systematic anti-generalizing associations.

### Phase 6 — identity-preserving Key–Value workspace

Added explicit slot identity, tied entity keys, tied value encoder/decoder, and query-conditioned top-1 retrieval. Exact and learned KV conditions reached 100% structural holdout accuracy in all 30 seeds. This established the necessary inductive bias.

### Phase 7 — position-independent sparse writes

Removed fixed position-to-slot addressing. First-free, hard learned, and annealed learned write allocation all reached 100%, but all collision-free permutations were behaviorally equivalent. The task could not identify whether routing was actually learned.

### Phase 8 — scarce retention and eviction

Made routing quality behaviorally consequential: six candidate bindings compete for three slots, with three context-relevant entities. The learned policy chooses write, skip, or eviction actions. Annealing succeeded while hard-from-start remained near heuristic baselines, directly supporting the routing credit-assignment hypothesis.

---

## 5. Phase 8 task and model

Each episode contains:

- one context token;
- six distinct entity-value bindings;
- exactly three context-relevant entities and three irrelevant entities;
- a three-slot workspace for all scarce conditions;
- a final query sampled from the relevant entities.

The model uses identity-preserving tied entity and value codebooks. The retention policy selects among skip, fill-empty, or evict-slot actions. The annealed condition uses straight-through hard forward actions with a soft surrogate during the first 900 of 1200 updates, then uses a strictly hard-only final 300-update window.

Conditions:

- `kv_full_capacity`
- `kv_oracle_retention`
- `kv_fifo_eviction`
- `kv_reservoir`
- `kv_hard_retention`
- `kv_annealed_retention`

Formal causal interventions:

- zero query context;
- randomized retention actions;
- forced FIFO;
- forced relevant eviction;
- permuted context labels;
- disabled skip;
- removed workspace writes;
- removed workspace persistence;
- removed KV output.

---

## 6. Phase 8 formal result

| Condition | Holdout NLL | Accuracy | Brier | Query hit | Relevant eviction |
|---|---:|---:|---:|---:|---:|
| Full capacity | 0.003761 | 100.000% | 0.000017 | 100.000% | 0.000% |
| Oracle retention | 0.003761 | 100.000% | 0.000017 | 100.000% | 0.000% |
| FIFO eviction | 1.357032 | 58.896% | 0.622601 | 50.563% | 24.750% |
| Reservoir | 1.336335 | 60.063% | 0.611034 | 51.813% | 15.056% |
| Hard from start | 1.366183 | 59.375% | 0.615963 | 51.104% | 12.323% |
| Annealed retention | **0.269222** | **94.833%** | **0.095143** | **94.042%** | **2.306%** |

Annealed retention significantly improves NLL, Brier, and accuracy over FIFO, reservoir, and hard-from-start in 30/30 seeds. The final 300-update disagreement rate is exactly zero.

All preregistered causal interventions are harmful in all 30 seeds. See `results/PHASE8_VALIDATION.md` for confidence intervals and exact effect sizes.

### Critical limitation

Convergence is bimodal:

- mean accuracy: 94.833%;
- median accuracy: 99.688%;
- minimum accuracy: 73.750%;
- 23/30 seeds reach at least 90%;
- 16/30 seeds reach at least 99%.

Accuracy correlates almost perfectly with query-hit rate (`0.99645`) and negatively with relevant-eviction rate (`-0.97821`). The next problem is optimization stability and delayed utility, not capacity.

---

## 7. Build and validation commands

### GCC Debug

```bash
cmake -S . -B build-gcc -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-gcc --parallel 2
ctest --test-dir build-gcc --output-on-failure
```

### Clang Debug

```bash
env CC=clang CXX=clang++ \
  cmake -S . -B build-clang -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-clang --parallel 2
ctest --test-dir build-clang --output-on-failure
```

### Release

```bash
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel 2
ctest --test-dir build-release --output-on-failure
```

### ASan/UBSan

```bash
env CC=clang CXX=clang++ cmake -S . -B build-sanitize -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DSGW_ENABLE_SANITIZERS=ON
cmake --build build-sanitize --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build-sanitize --output-on-failure
```

OpenAI host commands may be terminated near 120 seconds. Split configure, build, C++ tests, CLI tests, and Python aggregators into separate shell calls. Do not interpret an outer timeout as a test failure.

### Direct Phase 8 shard

```bash
scripts/run_phase8_shard.sh \
  build-release 0 5 1200 8 160 160 results/phase8-shard
```

### Aggregate a complete 30-seed result

```bash
python3 scripts/aggregate_phase8.py \
  --raw-dir results/merged-phase8/raw \
  --causal-dir results/merged-phase8/causal \
  --output-dir results/phase8-formal \
  --seed-start 0 --seed-count 30 \
  --bootstrap-replicates 50000

cd results/phase8-formal
sha256sum --check SHA256SUMS
```

### Trigger the formal workflow

The formal workflow is intentionally manual:

```bash
gh workflow run phase8-experiments.yml \
  --repo valurius38027/ESM \
  --ref agent/phase8-retention-eviction \
  -f steps=1200 -f seed_count=30
```

Do not add an automatic `push` or `pull_request` trigger merely for convenience; report and checksum commits must not rerun the expensive matrix.

---

## 8. Engineering invariants

Preserve these unless a new phase explicitly preregisters a change:

1. C++20 and standard-library-only runtime.
2. CPU-only formal experiments.
3. Deterministic output for a fixed seed and condition.
4. Stable index tie-breaking for hard selection.
5. LF-only CSV and JSON output.
6. Exact `steps * batch_size` online sample accounting.
7. Complete condition × seed Cartesian validation before aggregation.
8. No stale, duplicate, or extra shard files.
9. Every formal result directory includes a verified `SHA256SUMS`.
10. Causal interventions are evaluation-only and never retrain the model.
11. Final scientific reports distinguish capacity, optimization, causal dependence, and generalization.
12. Negative results are retained; do not rewrite history to present a monotonic success narrative.

---

## 9. Known technical debt and hazards

- GCC Release has previously emitted an optimizer-time `-Wnull-dereference` warning in the dataset enumeration path. Clang, Debug, ASan, and UBSan have not reproduced an actual invalid access. Investigate separately; do not suppress the warning globally.
- `src/model.cpp` is large. Avoid an opportunistic architectural refactor during the next experiment.
- The remote branch history contains materializer commits because GitHub connector uploads were used under a DNS-restricted container. The branch tree and formal outputs are canonical even when old local bundle commit hashes differ.
- Formal logs contain elapsed-time fields that vary by environment. Primary and causal CSVs must remain byte-identical for deterministic reruns; logs need not.
- The Phase 8 success is task-specific. Context relevance and tied KV identities are explicit. Do not claim general memory management or general intelligence.
- The branch is a draft PR and should remain unmerged until the user explicitly decides how to integrate the research line.

---

## 10. Immediate next actions

Use the companion plan:

`docs/superpowers/plans/2026-07-17-sgw-esm-phase9-delayed-utility-stability.md`

The correct order is:

1. reproduce Phase 8 from the handoff bundle with a small smoke matrix;
2. implement Phase 9A convergence diagnostics and stabilization on the unchanged Phase 8 task;
3. select a stabilization recipe using preregistered seed-robustness gates;
4. only then introduce Phase 9B variable delay and longer-horizon utility;
5. run formal paired Actions matrices and independently reaggregate artifacts;
6. commit a bounded report, whether the result is positive or negative.

Do not combine stabilization changes and delayed-task changes in one initial experiment. That would make failure attribution impossible.
