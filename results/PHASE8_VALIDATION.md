# SGW-ESM Phase 8 Validation

## Scope

Phase 8 tests retention and eviction under genuine workspace scarcity. Each episode contains six distinct entity-value bindings, only three workspace slots, and a context token that identifies the three potentially queried entities. The model must decide whether to skip an incoming binding, fill an empty slot, or evict an occupied slot.

The formal matrix compares:

- `kv_full_capacity`: six-slot capacity upper bound;
- `kv_oracle_retention`: three-slot oracle that stores only context-relevant bindings;
- `kv_fifo_eviction`: deterministic FIFO baseline;
- `kv_reservoir`: deterministic seeded reservoir baseline;
- `kv_hard_retention`: learned hard policy from the first update;
- `kv_annealed_retention`: straight-through soft-to-hard policy, annealed for 900 updates and evaluated with a final 300-update hard-only window.

All conditions share the identity-preserving entity/value codebooks and query-conditioned top-1 read path established in Phase 6.

## Execution and provenance

- GitHub Actions run: `29559003157`
- Source head: `0daf86d4717fd2f800aef1b58443efb228cbf51b`
- Artifact ID: `8398546609`
- Artifact ZIP SHA-256: `3bb581b4b104f29e836d4cdbce0cb519727466c8df17e408c3595e1e5744300c`
- Paired seeds: `0..29`
- Updates per condition: `1200`
- Batch size: `8`
- Online training samples per learned condition: `9600`
- Bootstrap replicates: `50000`
- Independent bootstrap replicates: `200000` with a distinct RNG and seed

The artifact contains 180 per-condition primary CSV files, 30 causal CSV files, and 180 logs. The consolidated tables contain 180 primary rows and 300 causal rows. Every raw row exactly matches the corresponding consolidated row.

The repository aggregator was rerun from the downloaded raw shards. All eight formal output files, including `SHA256SUMS`, were byte-identical to the Actions artifact.

## Formal results

| Condition | Holdout NLL | Holdout accuracy | Brier | Query hit | Relevant eviction |
|---|---:|---:|---:|---:|---:|
| Full capacity | 0.003761 | 100.000% | 0.00001698 | 100.000% | 0.000% |
| Oracle retention | 0.003761 | 100.000% | 0.00001698 | 100.000% | 0.000% |
| FIFO eviction | 1.357032 | 58.896% | 0.622601 | 50.563% | 24.750% |
| Reservoir | 1.336335 | 60.063% | 0.611034 | 51.813% | 15.056% |
| Hard from start | 1.366183 | 59.375% | 0.615963 | 51.104% | 12.323% |
| Annealed retention | **0.269222** | **94.833%** | **0.095143** | **94.042%** | **2.306%** |

The oracle and full-capacity conditions both reach 100% accuracy, proving that the three-slot task is solvable when retention decisions are correct.

### Annealed versus FIFO

- NLL improvement: `1.087810`, 95% paired bootstrap CI `[0.945795, 1.216362]`, favorable in 30/30 seeds;
- Brier improvement: `0.527458`, CI `[0.469027, 0.579408]`, favorable in 30/30 seeds;
- accuracy improvement: `0.359375`, CI `[0.322292, 0.392917]`, favorable in 30/30 seeds.

### Annealed versus reservoir

- NLL improvement: `1.067113`, CI `[0.934415, 1.189104]`, favorable in 30/30 seeds;
- Brier improvement: `0.515891`, CI `[0.462883, 0.564363]`, favorable in 30/30 seeds;
- accuracy improvement: `0.347708`, CI `[0.315417, 0.377292]`, favorable in 30/30 seeds.

### Annealed versus hard from start

- NLL improvement: `1.096960`, CI `[0.963883, 1.221871]`, favorable in 30/30 seeds;
- Brier improvement: `0.520820`, CI `[0.464694, 0.573713]`, favorable in 30/30 seeds;
- accuracy improvement: `0.354583`, CI `[0.314375, 0.393333]`, favorable in 30/30 seeds.

This comparison is the main credit-assignment result: the same learned policy class fails near the heuristic baselines when forced hard from the first update, but succeeds after straight-through temperature annealing.

## Hard-only tail

During the final 300 updates, the annealed model uses hard actions without soft/hard disagreement:

- query read hit rate: `94.133%`;
- relevant eviction rate: `2.433%`;
- write rate: `66.273%`;
- skip rate: `33.727%`;
- eviction rate: `16.273%`;
- final soft/hard disagreement rate: exactly `0` for every seed.

The retained performance therefore does not depend on leaving a soft policy active at inference time.

## Causal interventions

Every preregistered causal intervention harms the annealed policy in all 30 seeds.

| Intervention | NLL harm | Accuracy loss |
|---|---:|---:|
| Zero query context | 2.154452 | 38.563 pp |
| Randomize retention actions | 2.617365 | 56.354 pp |
| Force FIFO | 2.085235 | 37.396 pp |
| Force relevant eviction | 2.956260 | 63.813 pp |
| Permute context labels | 2.401583 | 53.250 pp |
| Disable skip | 1.441025 | 25.896 pp |
| Remove workspace writes | 1.522537 | 72.229 pp |
| Remove workspace persistence | 1.522537 | 72.229 pp |
| Remove mechanism output | 1.522537 | 72.229 pp |

The context, skip decision, eviction policy, persistent workspace, and KV output are all causally necessary. Removing the learned output no longer improves performance, unlike the anti-generalizing mediation path observed in Phase 5.

## Seed stability

The mean accuracy gate passes, but optimization remains bimodal across seeds:

- mean accuracy: `94.833%`;
- median accuracy: `99.688%`;
- minimum accuracy: `73.750%`;
- 23/30 seeds reach at least 90%;
- 16/30 seeds reach at least 99%.

Accuracy is tightly coupled to retention quality:

- query-hit versus accuracy correlation: `0.99645`;
- relevant-eviction versus accuracy correlation: `-0.97821`.

Phase 8 therefore establishes the capability and the annealing advantage, but not seed-robust convergence.

## Scientific verdict

All preregistered Phase 8 gates pass:

1. the three-slot task is solvable by the oracle;
2. annealed retention exceeds 90% mean holdout accuracy;
3. annealed retention significantly outperforms FIFO and reservoir on NLL, Brier, and accuracy;
4. the final 300-update policy is strictly hard;
5. query-hit and relevant-eviction quality gates pass;
6. every formal intervention causes significant degradation.

The defensible conclusion is:

> Under real workspace scarcity, routing quality becomes identifiable. Straight-through soft-to-hard annealing enables a sparse retention/eviction policy that learns context-dependent storage, substantially outperforming FIFO, reservoir, and the identical policy trained hard from the first update. The learned policy remains fully discrete in its final training window and inference path.

The result is still limited to a strongly structured synthetic task with an explicit context-relevance relation and tied key-value representations. It does not yet establish general-purpose memory management, variable-duration credit assignment, or autonomous expert routing. The residual seed failures indicate that the next phase should focus on convergence robustness and longer-horizon, delayed utility rather than adding more architectural freedom immediately.
