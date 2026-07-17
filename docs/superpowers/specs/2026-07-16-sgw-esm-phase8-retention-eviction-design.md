# SGW-ESM Phase 8 Scarce Workspace Retention and Eviction Design

## Objective

Phase 7 showed that any collision-free slot permutation is functionally equivalent when workspace capacity equals the number of bindings. Phase 8 makes write policy identifiable by presenting six bindings to a three-slot workspace. The model must retain context-relevant bindings, skip irrelevant bindings, or evict an existing slot.

The experiment isolates one question: can a sparse Key–Value workspace learn which information deserves scarce persistent capacity from final query loss?

## Task Distribution

Each episode uses:

- 3 context classes;
- 9 entity identities;
- 6 value classes;
- 6 distinct entity-value bindings;
- 3 workspace slots;
- one final query.

Context `c` defines the relevant entity set:

`R(c) = {e | e mod 3 == c}`

Every episode contains all three relevant entities and three uniformly sampled irrelevant entities. Binding order is deterministically shuffled. Values follow the existing structural holdout rule: training excludes `value == entity mod value_count`; the queried entity uses that held-out value in evaluation. The query is sampled uniformly from the three relevant entities.

The token sequence is:

`CONTEXT_c, E_0, V_0, ..., E_5, V_5, QUERY, E_q`

Context tokens are appended after the existing query token so all Phase 1–7 token IDs remain unchanged.

## Conditions

- `kv_full_capacity`: six slots, first-free writes, no retention pressure; capacity upper bound.
- `kv_oracle_retention`: three slots; writes relevant entities and skips irrelevant entities.
- `kv_fifo_eviction`: three slots; always writes and evicts the oldest slot when full.
- `kv_reservoir`: three slots; deterministic seeded reservoir sampling.
- `kv_hard_retention`: learned hard action from the first update.
- `kv_annealed_retention`: hard forward action with a straight-through soft surrogate for updates 0–899, then hard-only routing for updates 900–1199.

All conditions preserve Phase 6–7 tied entity/value codebooks, exact one-slot writes, query-conditioned top-1 reads, and CPU-only C++20 execution.

## Retention Actions

When a binding arrives, the action set is:

- `skip`;
- `write slot 0`;
- `write slot 1`;
- `write slot 2`.

Writing to an occupied slot is an eviction and replacement. Empty-slot writes are preferred only through the policy; no occupancy mask hides occupied actions.

For learned policies, normalized context code `C_c`, incoming entity code `K_e`, stored slot keys `K_j`, and normalized slot age `A_j` produce:

`r_in = dot(C_c, K_e)`

`r_j = dot(C_c, K_j)` for occupied slots, otherwise `-1`

`skip_logit = -r_in`

`write_logit_j = r_in - r_j + alpha * A_j`

`alpha` is a learned scalar. A stable argmax chooses the hard forward action. The annealed policy uses:

`g_a = one_hot(a*) + p_a - stop_gradient(p_a)`

with `p = softmax(logits / tau)`. Temperature decreases linearly from 2.0 to 0.1 through update 899. The final 300 updates use hard routing without a surrogate.

Entity and value tokens share the same retained action. A skip action writes neither key nor value. A write action replaces both key and value in the selected slot and resets its age.

## Deterministic Baselines

- Full capacity writes every binding into the first free slot.
- Oracle computes relevance from the public task rule and never evicts a relevant binding.
- FIFO writes every binding; when full, it replaces the greatest-age slot, breaking ties by lower slot index.
- Reservoir uses a deterministic hash of model seed, context, entity, and binding index. For stream index `t`, it replaces slot `j` only when `j = hash mod (t + 1)` is less than capacity; otherwise it skips.

## Telemetry

Each sequence and aggregate records:

- write, skip, and eviction counts;
- relevant writes and irrelevant writes;
- relevant evictions and irrelevant evictions;
- relevant retention rate at query time;
- queried-entity retention and read-hit rates;
- action entropy and hard/soft disagreement;
- per-slot write and eviction load;
- mean retained-slot age;
- final 300-update hard-only metrics.

`relevant eviction` means an occupied relevant entity was replaced. `queried-entity retention` is the primary causal diagnostic.

## Interventions

- `zero_query_context`: replace the context code by zero.
- `randomized_retention_actions`: cyclically shift the selected action.
- `force_fifo_retention`: replace learned actions with FIFO actions.
- `force_relevant_eviction`: when possible, evict a relevant occupied slot.
- `permuted_context_labels`: use the next context class.
- `disable_retention_skip`: replace skip with a deterministic write.
- existing `no_workspace_writes`, `no_workspace_persistence`, `permuted_workspace_keys`, `zero_query_key`, and `no_mechanism_output`.

## Formal Gates

For 30 paired seeds, 1200 updates, batch size 8, and 50,000 paired bootstrap replicates:

1. `kv_full_capacity` holdout accuracy is at least 0.99 in every seed.
2. `kv_oracle_retention` holdout accuracy is at least 0.99 in every seed.
3. `kv_annealed_retention` mean holdout accuracy is at least 0.90.
4. Annealed retention significantly exceeds FIFO and reservoir in NLL, Brier, and accuracy; all paired 95% intervals exclude zero.
5. Annealed queried-entity retention and read-hit rates are at least 0.90.
6. Annealed relevant-eviction rate in the final 300 updates is below 0.05.
7. The final 300 updates use hard one-hot actions only.
8. Every preregistered intervention significantly degrades the annealed model; removing KV output must not improve it.
9. `kv_hard_retention` is diagnostic. Annealed success with hard failure supports the routing-credit hypothesis; equal success means annealing is unnecessary.

## Boundaries

Success would establish learned retention and eviction under a fixed context rule, fixed six-binding episode length, and tied Key–Value representation. It would not establish autonomous context discovery, variable workspace size, natural-language memory selection, or long-horizon continual memory management.
