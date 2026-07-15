# SGW-ESM C++ Research Prototype

An independent C++20 research implementation of a **Sparse Global Workspace Emergent Sparse Model**.

The repository does not inherit code or Git history from the earlier Rust `esm_block_sparse` sandbox. Earlier experimental findings only motivate the controls used here.

## Current milestone

Phase 1 implements a deterministic, CPU-only, trainable minimum closure:

- a small always-on recurrent spine;
- persistent stateful mechanisms with learned hard top-k activation;
- a slot-based workspace with hard top-q competitive writes;
- recipient-selected hard top-m sparse broadcast;
- exact execution and mutation traces;
- explicit per-token operation estimates;
- scalar reverse-mode autodiff for auditable toy training;
- final-target-only delayed binding experiments;
- a true `core_only` ablation that allocates no unused SGW parameters.

The implementation is standard-library-only at runtime. CMake, a C++20 compiler and Ninja or Make are required to build it.

## Architecture

For each token:

1. the recurrent spine updates from the token and pooled previous workspace;
2. a learned router scores every mechanism and executes exactly `k`;
3. only active mechanism states are transformed;
4. active messages compete and exactly `q` write to workspace slots;
5. the updated workspace selects exactly `m` broadcast recipients;
6. non-active mechanism states and non-recipient inboxes remain exactly unchanged;
7. the final classifier reads the spine, workspace and active-mechanism summary.

Hard selection is piecewise differentiable. Gradients flow through selected computations and selected route weights; no straight-through estimator is used. Finite-difference checks discard perturbations that alter the exact route trace.

## Build and test

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Complete GCC, Clang, sanitizer and release gates are intentionally split so every host command remains below the two-minute ceiling:

```bash
timeout 110s scripts/run_validation.sh gcc
timeout 110s scripts/run_validation.sh clang
timeout 110s scripts/run_validation.sh sanitize
timeout 110s scripts/run_validation.sh release
```

Current validation status:

- 33 focused C++ tests;
- deterministic CLI integration test;
- LF-only deterministic result-consolidation regression test;
- GCC 14.2 warning-clean;
- Clang 17 warning-clean;
- AddressSanitizer and UndefinedBehaviorSanitizer clean;
- all CTest gates pass.

## Run an experiment

```bash
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release

./build-release/sgw_train \
  --mode sgw \
  --seed 0 \
  --steps 120 \
  --batch-size 6 \
  --train-count 96 \
  --holdout-count 32 \
  --output results/raw/sgw_seed0.csv
```

Available modes:

- `core_only`: embedding, recurrent spine and output head only;
- `sgw`: recurrent spine plus sparse mechanisms, workspace and sparse broadcast.

Formal five-seed reproduction:

```bash
timeout 110s scripts/run_formal.sh build-release
```

## Phase 1 findings

Five paired seeds on the controlled delayed binding task produced:

| Metric | Core only | SGW |
|---|---:|---:|
| Parameters | 327 | 2,182 |
| Estimated multiply-adds/token | 210 | 1,092 |
| Final train NLL | 0.7617 ± 0.1504 | 0.5994 ± 0.1094 |
| Final holdout NLL | 1.5186 ± 0.3969 | 1.4728 ± 0.2547 |
| Final holdout accuracy | 0.4375 ± 0.1344 | 0.4375 ± 0.0963 |

SGW trains and obeys its exact sparse budgets, but it has not demonstrated superiority. Holdout accuracy is equal, the small mean NLL difference changes direction across seeds, and SGW currently uses roughly `6.67x` the parameters and `5.20x` the estimated compute.

The defensible result is:

> The sparse global workspace loop is implementable, differentiable inside fixed-route regions and trainable without direct route labels. Its modeling or efficiency advantage remains an open experimental question.

See [`results/VALIDATION.md`](results/VALIDATION.md) for commands, hashes, evidence and limitations.

## Repository map

```text
include/sgw/autodiff.hpp       scalar reverse-mode autodiff API
include/sgw/config.hpp         model and optimizer invariants
include/sgw/dataset.hpp        identifiable delayed binding task
include/sgw/experiment.hpp     training and evaluation API
include/sgw/model.hpp          SGW model state, traces and public model API
include/sgw/optimizer.hpp      global-norm-clipped Adam
include/sgw/tensor.hpp         owned parameters and deterministic initialization
include/sgw/topk.hpp           deterministic sparse selection primitives
src/                           implementations
apps/sgw_train.cpp             bounded deterministic experiment CLI
tests/                         unit, gradient, mutation and integration gates
scripts/run_validation.sh      complete build/test matrix
scripts/run_formal.sh          paired five-seed experiment
scripts/summarize_runs.py      deterministic stdlib-only consolidation
results/VALIDATION.md          validation record and scientific boundaries
```

## Scope boundary

This is not yet a low-power production kernel or a natural-language model. Scalar autodiff intentionally prioritizes auditability over throughput. Any claim about power efficiency requires a later optimized inference backend and direct hardware measurements.

## Phase 2: fair controls and causal evaluation

Phase 2 adds two stronger recurrent controls and evaluation-only interventions.
For the fixed delayed-binding task:

| Preset | Parameters | Estimated MAdds/token | Purpose |
|---|---:|---:|---|
| `core_small` | 327 | 210 | low-cost Phase 1 reference |
| `core_compute_matched` | 2,232 | 1,092 | exact active-compute match to SGW |
| `core_param_matched` | 2,181 | 1,474 | parameter match to one scalar |
| `sgw` | 2,182 | 1,092 | sparse global workspace model |

No integer-valued embedding/spine dimensions match both SGW constraints exactly,
so positive SGW claims must survive both the compute-matched and
parameter-matched controls.

A trained SGW can be evaluated without retraining under:

- `no_broadcast`;
- `no_workspace_persistence`;
- `no_workspace_output`;
- `permuted_recipients`.

The formal Actions workflow runs 30 paired seeds in six parallel shards:

```bash
scripts/run_phase2_shard.sh build-release 0 5 400 8 160 48 results/phase2-shard
python3 scripts/aggregate_phase2.py \
  --raw-dir results/phase2-shard/raw \
  --causal-dir results/phase2-shard/causal \
  --output-dir results/phase2-formal \
  --seed-start 0 --seed-count 5 --bootstrap-replicates 50000
```

The repository does not claim a Phase 2 advantage until the formal artifact is
complete and its paired confidence intervals satisfy the gates in
[`docs/superpowers/specs/2026-07-15-sgw-esm-phase2-controls-design.md`](docs/superpowers/specs/2026-07-15-sgw-esm-phase2-controls-design.md).
