# SGW-ESM Phase 9 Validation: Delayed Utility and Long-Horizon Credit Assignment

## 1. Scope

Phase 9 tests whether the discrete scarce-workspace retention policy established in Phase 8 can preserve useful bindings across a substantially longer interval of later write opportunities.

The primary episode contains:

- one context token;
- six candidate entity-value bindings, of which three are context-relevant;
- eighteen context-irrelevant distractor bindings;
- one final query;
- three persistent Key-Value workspace slots.

The resulting formal sequence has 51 tokens and a minimum source-to-query distance of 38 tokens. Every distractor passes through the same discrete skip/write/evict action path as the original candidate bindings. The model receives no distractor label.

The primary evaluation delay is 18 distractor bindings. Additional evaluations use delays 0, 6, 12, and 24.

## 2. Conditions

| Condition | Retention policy | Parameters |
|---|---|---:|
| `kv_delayed_oracle` | retains exactly the context-relevant candidates | 120 |
| `kv_delayed_fifo` | deterministic FIFO replacement | 120 |
| `kv_delayed_reservoir` | deterministic seeded reservoir baseline | 120 |
| `kv_delayed_hard` | learned policy, hard from the first update | 145 |
| `kv_delayed_annealed_direct` | straight-through annealing, always trained at delay 18 | 145 |
| `kv_delayed_annealed_curriculum` | straight-through curriculum over delays 0/6/12/18 | 145 |

The curriculum allocates 15%, 15%, 20%, and 50% of the 1,800 updates to delays 0, 6, 12, and 18. Annealing ends at update 1,350. The final 450 updates are hard-only at delay 18.

## 3. Formal protocol

- 30 paired seeds, `0..29`;
- 1,800 updates per condition;
- batch size 8;
- 14,400 unique online samples per learned condition;
- 160 diagnostic training samples;
- 160 structural holdout samples;
- 50,000 paired bootstrap replicates;
- independent 200,000-replicate bootstrap with a different RNG;
- six GitHub Actions shards.

GitHub Actions run: `29567182862`
Source head: `30858cffcb2c3df6b57600d982d68fcad68e747e`
Artifact: `8401680872`
Artifact SHA-256: `cc4670690b38396eefb32179d2548798d444d1de726bf51cd31b2c26ac831e59`

The artifact contains:

- 180 primary condition CSV files;
- 30 causal CSV files containing 390 intervention rows;
- 180 logs;
- nine consolidated formal files.

The repository aggregator regenerated all nine formal files byte-for-byte from the downloaded raw data. All committed files pass `SHA256SUMS`.

## 4. Primary results at delay 18

| Condition | NLL | Accuracy | Brier | Query hit | Relevant survival | Distractor write |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | 0.003764 | 100.000% | 0.000017 | 100.000% | 100.000% | 0.000% |
| FIFO | 1.792423 | 20.375% | 0.833506 | 0.000% | 0.000% | 100.000% |
| Reservoir | 1.763647 | 29.375% | 0.820683 | 11.688% | 12.056% | 22.469% |
| Hard from start | 1.634890 | 41.833% | 0.752631 | 29.646% | 29.840% | 38.684% |
| Annealed direct | 1.090244 | 68.604% | 0.469242 | 61.125% | 61.167% | 32.500% |
| **Annealed curriculum** | **0.910743** | **78.229%** | **0.379427** | **72.604%** | **72.597%** | **24.671%** |

The oracle remains perfect at delays 18 and 24. The task is representationally solvable and the persistent workspace itself does not decay with sequence length.

## 5. Paired comparisons

### Curriculum versus FIFO

- NLL improvement: `0.881680`, 95% CI `[0.743771, 1.041981]`, favorable in 30/30 seeds;
- Brier improvement: `0.454079`, CI `[0.396103, 0.520856]`, favorable in 30/30;
- accuracy improvement: `57.854` percentage points, CI `[54.354, 61.812]`, favorable in 30/30.

### Curriculum versus reservoir

- NLL improvement: `0.852904`, CI `[0.715521, 1.011635]`, favorable in 30/30;
- Brier improvement: `0.441256`, CI `[0.383786, 0.507960]`, favorable in 30/30;
- accuracy improvement: `48.854` percentage points, CI `[45.208, 52.875]`, favorable in 30/30.

### Curriculum versus hard from start

- NLL improvement: `0.724147`, CI `[0.606616, 0.861300]`, favorable in 30/30;
- Brier improvement: `0.373204`, CI `[0.322801, 0.430695]`, favorable in 30/30;
- accuracy improvement: `36.396` percentage points, CI `[31.792, 41.083]`, favorable in 30/30.

### Curriculum versus direct annealing

- accuracy improvement: `9.625` percentage points, CI `[1.938, 17.646]`, favorable in 22/30 seeds;
- query-hit improvement: `11.479` points, CI `[1.750, 21.563]`;
- relevant-survival improvement: `11.431` points, CI `[1.799, 21.625]`;
- NLL improvement: `0.179501`, CI `[-0.046627, 0.398488]`;
- Brier improvement: `0.089815`, CI `[-0.008212, 0.185715]`.

The curriculum has a statistically separated advantage in accuracy and retention quality, but not in NLL or Brier under the preregistered 95% interval.

### Direct annealing versus hard from start

Direct annealing also improves over hard-from-start:

- NLL: `0.544646`, CI `[0.385463, 0.719118]`;
- Brier: `0.283389`, CI `[0.207442, 0.362854]`;
- accuracy: `26.771` points, CI `[18.417, 34.792]`.

Thus the Phase 8 credit-assignment result survives the longer task. Annealing remains necessary, and the delay curriculum adds a further retention-quality benefit.

## 6. Delay curve

Curriculum performance is:

| Delay bindings | Accuracy | Query hit | Relevant survival |
|---:|---:|---:|---:|
| 0 | 81.854% | 77.896% | 77.833% |
| 6 | 78.250% | 72.438% | 72.444% |
| 12 | 77.250% | 71.625% | 71.639% |
| 18 | 78.229% | 72.604% | 72.597% |
| 24 | 77.542% | 71.604% | 71.611% |

The principal drop occurs between zero and six distractors. Performance is then approximately flat through delay 24. This does not resemble progressive memory decay with temporal distance. It indicates convergence to a partially selective retention policy that repeatedly admits some irrelevant bindings.

FIFO provides the complementary control: its query hit falls from 50.56% at delay 0 to exactly zero at every nonzero delay. Reservoir degrades monotonically with delay. The learned curriculum policy therefore carries useful information across the full horizon, but does not learn sufficiently selective distractor rejection.

## 7. Seed stability and failure mode

The preregistered long-horizon capability gate fails:

- mean delay-18 accuracy: `78.229%`, required `>=90%`;
- seeds at or above 90%: `5/30`, required `>=27/30`;
- minimum accuracy: `67.500%`, required `>=80%`;
- mean delay-24 accuracy: `77.542%`, required `>=85%`;
- mean delay-18 query hit: `72.604%`, required `>=90%`;
- mean delay-18 relevant survival: `72.597%`, required `>=90%`.

Five seeds—`2, 15, 18, 22, 29`—learn a perfect policy. Their distractor write rate is zero. The other 25 seeds form a broad partial-policy cluster:

- mean accuracy: `73.875%`;
- mean query hit: `67.125%`;
- mean distractor write rate: `29.606%`;
- mean relevant eviction rate: `3.467%`.

Across seeds:

- correlation of accuracy with query hit: `0.99220`;
- correlation of accuracy with relevant survival: `0.99223`;
- correlation of accuracy with distractor write rate: `-0.95978`;
- correlation of accuracy with relevant eviction rate: `-0.91279`.

The dominant failure is therefore repeated admission of irrelevant distractors, not a failure to decode retained values or an unavoidable decay of stored representations.

## 8. Hard-only and causal checks

The final 450 updates are fully discrete. Soft/hard disagreement is exactly zero in all 30 curriculum seeds.

Every preregistered causal intervention is harmful with a positive 95% lower bound for both NLL and accuracy. Accuracy losses include:

- zero query context: `51.500` points;
- randomized retention actions: `42.875` points;
- forced relevant eviction: `56.813` points;
- no workspace writes: `55.229` points;
- no workspace persistence: `55.229` points;
- no mechanism output: `55.229` points;
- relevant-looking delay distractors: `20.792` points.

Two diagnostic interventions are intentionally not part of the causal pass gate:

- removing delay distractors improves accuracy by `3.188` points;
- reversing the delay block changes accuracy by only `-0.250` points, with an interval spanning zero.

This is consistent with a content-selectivity bottleneck rather than sensitivity to distractor ordering.

## 9. Gate outcome

| Gate | Result |
|---|---|
| Oracle capacity | PASS |
| Hard-only final window | PASS |
| Curriculum versus heuristic/hard baselines | PASS |
| Curriculum advantage over direct training | PASS |
| Causal necessity | PASS |
| **Long-horizon capability and seed robustness** | **FAIL** |
| **Overall Phase 9 gate** | **FAIL** |

## 10. Scientific conclusion

Phase 9 does not support the claim that the present retention learner has solved stable long-horizon memory management.

It does support four narrower conclusions:

1. The Key-Value workspace can preserve exact bindings over at least 24 later write opportunities when the retention policy is correct.
2. Straight-through annealing remains substantially better than hard-from-start training under long delayed utility.
3. A delay curriculum improves accuracy, query hit, and relevant survival over direct long-delay training.
4. The remaining limitation is a bimodal policy-learning failure: a minority of seeds learn perfect distractor rejection, while most settle into a partially selective policy that repeatedly writes irrelevant bindings.

The next phase should target policy convergence and selective write suppression rather than increasing delay length further. A defensible Phase 10 would introduce an explicit sparse write budget or dual/Lagrangian write-cost constraint, while preserving the final task loss as the primary objective. The test should determine whether write regularization converts the 25 partial-policy seeds into the perfect-rejection basin without damaging useful writes.

## 11. Verification notes

- 93 focused C++ tests pass locally under GCC and Clang.
- GCC Release and deterministic CLI gates pass.
- Local LeakSanitizer crashes before `main()` even for a trivial independent probe; ASan/UBSan with leak detection disabled passes all four deterministic shards and the Phase 9 CLI.
- GitHub Actions run `29567182897` passes GCC Debug, Clang Debug, GCC Release, and ASan/UBSan with normal leak detection enabled, confirming the local failure is host-runtime-specific.
- The formal workflow is manual after the initial run to prevent result and documentation commits from repeating the matrix.
