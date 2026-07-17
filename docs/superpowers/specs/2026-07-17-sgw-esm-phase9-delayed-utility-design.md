# SGW-ESM Phase 9 Delayed Utility and Long-Horizon Credit Design

## Objective

Phase 9 tests whether the discrete retention policy established in Phase 8 can preserve useful bindings across a substantially longer sequence of later write opportunities, and whether a delay curriculum improves convergence robustness without changing the final hard inference policy.

The experiment isolates temporal credit distance. It preserves the Phase 8 three-slot workspace, context-to-entity relevance rule, tied Key-Value representation, action space, and final query objective.

## Task distribution

Each episode contains:

1. one context token;
2. six candidate bindings, exactly three context-relevant and three context-irrelevant;
3. a configurable delay block containing context-irrelevant distractor bindings;
4. one query marker and one queried entity token.

The queried entity is sampled from the three relevant candidate bindings. Delay distractors are never query targets. They use context-irrelevant entity IDs and independently sampled non-holdout values. Repeated distractor entities are allowed, but the full token sequence must remain unique inside each generated split or stream.

Formal training uses 18 distractor bindings. The queried source value is therefore separated from the final query by between 39 and 49 tokens depending on its candidate position, compared with 3 to 13 tokens in Phase 8.

Evaluation uses delay lengths 0, 6, 12, 18, and 24. Delay 18 is the in-distribution primary endpoint. Delay 24 is the extrapolation endpoint.

## Conditions

- `kv_delayed_oracle`: stores relevant candidates and skips all distractors.
- `kv_delayed_fifo`: deterministic FIFO baseline.
- `kv_delayed_reservoir`: deterministic seeded reservoir baseline.
- `kv_delayed_hard`: learned retention policy using hard actions from update zero.
- `kv_delayed_annealed_direct`: straight-through annealing trained directly on delay 18.
- `kv_delayed_annealed_curriculum`: same model and optimizer; training delay schedule is 0 for the first 15% of updates, 6 for the next 15%, 12 for the next 20%, and 18 for the final 50%. This gives delay 18 exactly 450 surrogate-gradient updates before the final 450 hard-only updates.

The two annealed conditions have identical model parameters, temperature schedule, optimizer, sample budget, and final hard-only duration. Only the task-delay schedule differs.

## Model behavior

The Phase 8 retention action space is unchanged:

- `skip`;
- write an empty slot;
- evict slot 0, 1, or 2 and write the incoming binding.

The model receives the context, incoming key, occupied slot keys, and slot ages. Delay distractors are processed through the same policy path as candidate bindings. There is no special model-side distractor flag.

The forward path accepts a variable number of binding pairs and validates the sequence as:

`context + N*(entity,value) + query_marker + query_entity`, where `N >= 6`.

The first six binding pairs are candidates; remaining pairs are delay distractors. Dataset metadata records candidate count, delay count, source-to-query token distance, and whether each binding is a delay distractor.

## Training protocol

Formal runs use:

- 30 paired seeds;
- 1,800 updates;
- batch size 8;
- 14,400 unique online samples per learned condition;
- 160 train diagnostics and 160 holdout samples per evaluation delay;
- temperature annealing for the first 1,350 updates;
- final 450 updates in strict hard-only mode;
- 50,000 paired bootstrap replicates in the formal aggregator;
- an independent 200,000-replicate bootstrap implementation.

All conditions use the same entity/value/context vocabulary and the same paired dataset seeds.

## Metrics

Existing NLL, accuracy, Brier, ECE, query-hit, relevant-eviction, write, skip, eviction, action-entropy, disagreement, and slot-load metrics remain.

Phase 9 adds:

- `delay_binding_count`;
- `mean_source_query_distance`;
- `distractor_write_rate`;
- `distractor_eviction_rate`;
- `relevant_survival_rate_after_delay`;
- `query_hit_by_delay` for 0, 6, 12, 18, and 24;
- `accuracy_by_delay` and `nll_by_delay`;
- final-450 hard-only counterparts for distractor write, relevant survival, and query hit.

## Causal interventions

The primary curriculum model is evaluated with:

- zero context;
- randomized retention actions;
- forced FIFO;
- forced relevant eviction;
- permuted context labels;
- disabled skip;
- no workspace writes;
- no workspace persistence;
- no Key-Value output;
- remove delay distractors;
- replace delay distractors with relevant-looking keys;
- reverse the delay block.

The last three interventions distinguish dependence on temporal interference from dependence on candidate selection alone.

## Preregistered gates

Capacity gate:

- delayed oracle accuracy at delay 18 is at least 99%;
- delayed oracle accuracy at delay 24 is at least 99%.

Long-horizon capability gate for `kv_delayed_annealed_curriculum`:

- mean delay-18 accuracy is at least 90%;
- at least 27 of 30 seeds reach 90% delay-18 accuracy;
- minimum delay-18 accuracy is at least 80%;
- mean delay-24 accuracy is at least 85%;
- delay-18 query-hit rate is at least 90%;
- delay-18 relevant survival is at least 90%;
- delay-18 relevant-eviction rate is below 5%;
- final-450 soft/hard disagreement is exactly zero for every seed.

Comparative gate:

- curriculum beats FIFO, reservoir, and hard-from-start on NLL, Brier, and accuracy with paired 95% intervals excluding zero;
- curriculum beats direct-delay annealing on at least two of NLL, Brier, accuracy, query hit, or seed success count, without being significantly worse on any primary metric.

Causal gate:

- zero context, randomized actions, forced relevant eviction, no writes, no persistence, and no KV output are harmful in all 30 paired seeds;
- relevant-looking delay distractors are significantly more harmful than intact delay distractors.

## Interpretation boundaries

Passing establishes discrete long-horizon retention under controlled context relevance and sparse capacity. It does not establish arbitrary episodic memory, unbounded sequence generalization, or autonomous discovery of the relevance relation.

If oracle passes but both learned delayed conditions fail, the representation remains sufficient and the failure is assigned to long-horizon credit or optimization. If curriculum outperforms direct annealing, the result supports temporal curriculum as a convergence mechanism. If delay-18 passes but delay-24 collapses, the result is length-specific rather than algorithmic generalization.
