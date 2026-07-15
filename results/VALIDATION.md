# SGW-ESM C++ Phase 1 Validation

## 1. Validation status

Phase 1 is implemented as an independent C++20 repository. It does not inherit source code or Git history from the earlier Rust ESM sandbox.

Validated components:

- deterministic stable top-k selection and selected-set softmax;
- scalar reverse-mode autodiff with external parameter-gradient accumulation;
- validated parameter ownership, Xavier initialization and global-norm-clipped Adam;
- identifiable delayed binding data with unique, strictly disjoint train/holdout literal sequences;
- always-on recurrent spine;
- hard top-k stateful mechanism execution;
- hard top-q workspace write competition;
- fixed slot workspace with gated recurrent writes;
- hard top-m recipient-selected broadcast;
- exact mutation boundaries for inactive mechanisms and non-recipient inboxes;
- per-step route and estimated-operation traces;
- final-target cross entropy, BPTT and deterministic shuffling;
- finite-difference gradient checks that reject coordinates crossing an exact route boundary;
- `core_only` ablation that allocates only parameters it actually executes;
- deterministic command-line experiment output.

## 2. Environment

Recorded in `results/evidence/toolchain.txt`:

```text
2026-07-15T16:43:53Z
g++ (Debian 14.2.0-19) 14.2.0
clang version 17.0.0
cmake version 3.31.6
ninja 1.12.1
Python 3.13.5
```

Every validation subcommand was wrapped in `timeout 110s` where it could perform substantial work.

## 3. Build and test gates

The following configurations passed:

| Configuration | Result |
|---|---:|
| GCC 14.2 Debug, warnings enabled | pass |
| Clang 17 Debug, warnings enabled | pass |
| GCC Debug + AddressSanitizer + UndefinedBehaviorSanitizer | pass |
| GCC 14.2 Release | pass |

CTest contains three gates:

1. `sgw_tests`: 24 focused C++ tests;
2. `sgw_cli_determinism`: two identical CLI runs followed by byte-for-byte CSV comparison;
3. `sgw_summary_lf_endings`: self-contained consolidation fixture proving committed CSVs use LF-only line endings.

All four build configurations report 3/3 CTest gates passing. The unit harness reports:

```text
Executed 24 tests; 0 failed.
```

Evidence files:

- `results/evidence/unit_tests.log`
- `results/evidence/ctest_gcc.log`
- `results/evidence/ctest_clang.log`
- `results/evidence/ctest_sanitize.log`
- `results/evidence/ctest_release.log`
- `results/evidence/validation_gcc.log`
- `results/evidence/validation_clang.log`
- `results/evidence/validation_sanitize.log`
- `results/evidence/validation_release.log`

## 4. Gradient validation

The finite-difference gate:

1. performs one analytic backward pass through the complete SGW model;
2. scans deterministic nonzero parameter coordinates;
3. evaluates `+epsilon` and `-epsilon` perturbations;
4. rejects a coordinate when active mechanisms, writers, target slots or recipients differ from the base trace;
5. applies a mixed absolute/relative tolerance to accepted coordinates.

At least four stable, nonzero coordinates are required; the current test validates eight. This establishes local derivative correctness inside a fixed hard-route region. It does not define a derivative at route boundaries.

## 5. Formal Phase 1 smoke experiment

### Configuration

- paired seeds: `0, 1, 2, 3, 4`;
- modes: `core_only`, `sgw`;
- 120 optimization steps;
- batch size 6;
- 96 unique training sequences;
- 32 unique, disjoint holdout sequences;
- sequence length 10;
- five target classes;
- chance NLL: `ln(5) = 1.609437912434`;
- primary loss: final target cross entropy only;
- no router, writer, slot or recipient labels;
- SGW hard budgets per token: 2 active mechanisms, 1 writer, 2 recipients.

Raw shards are generated under `results/raw/` and intentionally ignored by Git. The committed consolidated evidence is:

- `results/formal_runs.csv`
- `results/formal_summary.csv`
- `results/paired_results.csv`

### Aggregate results

| Metric | Core only | SGW |
|---|---:|---:|
| Parameters | 327 | 2,182 |
| Estimated multiply-adds/token | 210 | 1,092 |
| Initial holdout NLL, mean ± SD | 1.6745 ± 0.0467 | 1.6419 ± 0.0301 |
| Final train NLL, mean ± SD | 0.7617 ± 0.1504 | 0.5994 ± 0.1094 |
| Final holdout NLL, mean ± SD | 1.5186 ± 0.3969 | 1.4728 ± 0.2547 |
| Final holdout accuracy, mean ± SD | 0.4375 ± 0.1344 | 0.4375 ± 0.0963 |
| First 10-step batch loss | 1.6210 ± 0.0109 | 1.6552 ± 0.0255 |
| Last 10-step batch loss | 0.8162 ± 0.1169 | 0.6744 ± 0.1141 |

The paired quantity `core_only final holdout NLL - SGW final holdout NLL` has:

- mean: `0.045824`;
- SD: `0.213451`;
- standard error: `0.095458`.

The direction reverses across seeds. Five seeds are insufficient for a credible superiority claim.

### Interpretation

The experiment establishes the intended Phase 1 milestone:

- the SGW loop trains end-to-end without direct route supervision;
- training loss falls substantially;
- all hard budgets remain exact;
- route and mutation invariants remain valid;
- deterministic reruns reproduce identical consolidated results.

It does **not** establish an architectural advantage:

- mean holdout accuracy is identical in the two modes;
- paired holdout NLL differences are small relative to seed variance;
- SGW uses about `6.67x` the parameters;
- SGW uses `5.20x` the estimated multiply-adds per token;
- the current scalar autodiff backend is designed for auditability, not energy efficiency.

The correct conclusion is therefore: **the sparse global workspace minimum closure is implementable and trainable, but its scientific advantage remains unproven.**

## 6. Determinism evidence

The formal experiment was rerun from the release binary. Consolidated output hashes were unchanged:

```text
855f59c9e8a5e874ef407e7993a592dd5438b6f995ba71ee9dc5ab77784898a2  results/formal_runs.csv
966d6e474d397f72d96d3a1144fc54b1c190166af7c8d5fc4d49e6d05f3a65dd  results/formal_summary.csv
2cf4f2029c01f0597cc085eed5e99b6d9839dd29a5b16bb99ac9640d130e042e  results/paired_results.csv
```

The before/after records are retained in:

- `results/evidence/formal_hashes_before.txt`
- `results/evidence/formal_hashes_after.txt`

## 7. Reproduction

Full build gates are run as four independent commands so the host never has to keep one process alive for the complete matrix:

```bash
timeout 110s scripts/run_validation.sh gcc
timeout 110s scripts/run_validation.sh clang
timeout 110s scripts/run_validation.sh sanitize
timeout 110s scripts/run_validation.sh release
```

Formal paired smoke experiment:

```bash
timeout 110s scripts/run_formal.sh build-release
```

Single experiment:

```bash
timeout 110s ./build-release/sgw_train \
  --mode sgw \
  --seed 0 \
  --steps 120 \
  --batch-size 6 \
  --train-count 96 \
  --holdout-count 32 \
  --output results/raw/sgw_seed0.csv
```

## 8. Remaining scientific gates

Phase 2 must not merely enlarge the present model. It must add:

1. a parameter-matched recurrent baseline;
2. an active-compute-matched dense or shared-mechanism baseline;
3. tasks that require separable submechanisms and cross-mechanism coordination;
4. workspace ablations: clearing, slot permutation, delayed broadcast and wrong-recipient broadcast;
5. route-specialization intervention tests across seeds;
6. an optimized dense-kernel inference backend with measured latency, bytes read, cache misses and joules/token;
7. larger paired seed counts and prespecified stopping criteria.

Until those gates pass, the repository is a controlled architecture laboratory rather than evidence for a superior low-power language model.
