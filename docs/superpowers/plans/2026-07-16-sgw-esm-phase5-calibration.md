# SGW-ESM Phase 5 Implementation Plan

**Goal:** Separate structural generalization and calibration failures from workspace mediation capacity.

**Stack:** C++20, CMake, Python standard library, GitHub Actions, CPU only.

- [x] Add deterministic structural entity-value holdout.
- [x] Add no-replacement online training stream with exact consumption telemetry.
- [x] Add parameter-free bounded output logits.
- [x] Add NLL, accuracy, Brier, ECE, confidence and true-class probability.
- [x] Add stream-training overload while preserving fixed-corpus behavior.
- [x] Extend deterministic CLI and causal CSV audit fields.
- [x] Add strict Phase 5 aggregator and regression gate.
- [x] Add six-shard formal experiment runner.
- [ ] Pass remote GCC, Clang, Release and ASan/UBSan gates.
- [ ] Run 30 paired seeds at 800 steps.
- [ ] Independently reaggregate and commit verified formal results.
