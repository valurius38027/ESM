# SGW-ESM Phase 4 Mediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Build and formally test an information-separated fixed sparse workspace mediation path.

**Architecture:** Add content-blind/full recurrent controls and a fixed writer-slot-reader SGW topology. Preserve exact route traces, add mediation-specific causal interventions and auxiliary-withdrawal telemetry, then run a 30-seed sharded Actions experiment.

**Tech Stack:** C++20, CMake, Ninja, Python standard library, GitHub Actions, CPU only.

## Global Constraints

- Runtime remains standard-library-only.
- Existing Phase 1-3 behavior and parameter counts remain unchanged.
- Formal results must be deterministic and checksum-verified.
- Workspace auxiliary weight must be exactly zero during the final 200 updates.

### Task 1: Add Phase 4 presets and invariants
- [x] Add content-access flags and fixed mediation configuration.
- [x] Add four experiment conditions and three model presets.
- [x] Test invalid bypasses and exact dimensions.

### Task 2: Add fixed sparse mediation path
- [x] Route the three binding pairs to fixed writers and slots.
- [x] Route both query tokens to the reader.
- [x] Make the reader the sole broadcast recipient.
- [x] Remove learned routing/address parameters from this preset.

### Task 3: Add task and withdrawal telemetry
- [x] Use the 8-token three-binding task for Phase 4 conditions.
- [x] Persist anneal-boundary primary loss, final auxiliary weight, and final weighted auxiliary contribution.

### Task 4: Add causal interventions
- [x] Add `no_workspace_writes` and `zero_reader_inbox`.
- [x] Preserve route traces while severing the intended information path.

### Task 5: Add deterministic formal pipeline
- [x] Add strict Phase 4 aggregation and regression test.
- [x] Add sharded runner and six-shard Actions workflow.

### Task 6: Validate and execute
- [ ] Pass GCC, Clang, ASan/UBSan, and Release gates.
- [ ] Publish a draft Phase 4 research PR.
- [ ] Run 30 paired seeds and independently reaggregate the artifact.
- [ ] Commit a scientifically bounded validation report.
