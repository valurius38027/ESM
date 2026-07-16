# Phase 7 Validation: Position-Independent Sparse Writes

## Scope

Phase 7 removes the Phase 6 fixed-position write rule while preserving the three-slot key-value workspace, tied entity/value codebooks, query-conditioned top-1 reads, structural entity-value holdout, and no-replacement training stream.

Five paired conditions were evaluated for 30 seeds, 800 updates, batch size 8, 160 training examples, and 160 holdout examples:

- `structural_core_blind`
- `kv_fixed_position`
- `kv_first_free`
- `kv_hard_router`
- `kv_annealed_router`

The formal aggregator used 50,000 paired bootstrap replicates. An independent implementation used a different RNG and 200,000 paired bootstrap replicates.

## Engineering validation

- GCC Debug: 74/74 tests passed.
- Clang 17 Debug: 74/74 tests passed.
- GCC Release built successfully; the pre-existing GCC optimizer warning in `dataset.cpp` remains.
- ASan/UBSan: four deterministic shards covered all 74 tests with zero failures.
- Sanitized CLI and causal paths passed.
- 150 formal condition rows and 450 causal rows were accepted by the strict Cartesian-product validator.
- Every committed formal file passes `SHA256SUMS`.

## Formal results

| Condition | Parameters | Holdout NLL | Holdout accuracy | Brier | ECE | Final-200 collision rate |
|---|---:|---:|---:|---:|---:|---:|
| Content-blind core | 725 | 1.647312 | 17.375% | 0.814091 | 0.054082 | 0 |
| Fixed-position KV | 88 | 0.002254 | 100% | 0.00000638 | 0.002251 | 0 |
| First-free KV | 88 | 0.002254 | 100% | 0.00000638 | 0.002251 | 0 |
| Hard learned-slot KV | 112 | 0.002254 | 100% | 0.00000638 | 0.002251 | 0 |
| Annealed learned-slot KV | 112 | 0.002254 | 100% | 0.00000638 | 0.002251 | 0 |

Annealed routing versus the content-blind core:

- NLL improvement: 1.645058, 95% CI [1.639535, 1.650797], favorable in 30/30 seeds.
- Brier improvement: 0.814085, 95% CI [0.812187, 0.816048], favorable in 30/30 seeds.
- Accuracy improvement: 82.625 percentage points, 95% CI [81.813, 83.458], favorable in 30/30 seeds.

The independent 200,000-replicate bootstrap reproduced the direction and separation of all three intervals.

## Causal validation

For the annealed condition, every preregistered intervention was harmful in 30/30 seeds:

| Intervention | NLL degradation | Accuracy loss |
|---|---:|---:|
| Randomized write slots | 2.683270 | 69.333 pp |
| Cleared writer assignment | 1.607184 | 66.375 pp |
| Forced write collisions | 1.014651 | 41.833 pp |
| Permuted workspace keys | 5.731873 | 76.542 pp |
| Zero query key | 3.793553 | 50.688 pp |
| Removed KV output | 1.607184 | 66.375 pp |

The workspace write/read path is therefore causally necessary under this task.

## Scientific decision

The capacity and position-independence gates pass:

1. First-free allocation reaches 100% in all seeds, proving fixed binding position is unnecessary.
2. Hard and annealed routers remain collision-free and achieve 100% structural holdout accuracy.
3. Write-slot usage is exactly balanced across the three slots.
4. Key, query, assignment, collision, and output interventions all destroy performance.

However, the router-credit-assignment claim is **not identified**. Fixed-position, first-free, hard-from-start, and annealed routing have exactly identical NLL, Brier, ECE, and accuracy for every one of the 30 seeds. Any collision-free slot permutation is functionally equivalent because keys and values travel together and query-time retrieval matches by key. The task rewards valid allocation but does not reward one learned allocation policy over another.

The defensible conclusion is:

> Phase 7 proves that the key-value workspace generalizes with position-independent, collision-free sparse allocation. It does not prove that slot routing was learned or that annealing solved routing credit assignment.

The next experiment must make allocation quality behaviorally consequential, for example by using fewer slots than candidate bindings, overwrite/eviction decisions, variable-length episodes, or a write-cost budget. Only then can hard-from-start and annealed routing be distinguished.
