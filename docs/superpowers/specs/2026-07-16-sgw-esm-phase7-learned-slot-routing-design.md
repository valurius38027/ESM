# SGW-ESM Phase 7 Learned Slot Routing Design

## Objective

Remove Phase 6's fixed binding-position-to-slot mapping while preserving its identity-safe entity keys, tied value codec, query-conditioned top-1 read, structural holdout, and sparse CPU-only execution.

Phase 7 isolates one question: can a writer choose a collision-free workspace slot without using binding position as the address?

## Conditions

- `kv_fixed_position`: Phase 6 learned KV reference; binding index selects slot.
- `kv_first_free`: deterministic first available slot; proves position-independent capacity.
- `kv_hard_router`: learned slot codebook with hard top-1 from the first update; diagnostic for credit-assignment failure.
- `kv_annealed_router`: straight-through hard forward route with softmax surrogate, temperature annealed during the first 600 of 800 updates and hard-only for the final 200 updates.

All four conditions use three bindings, three slots, randomized binding order, tied entity/value codebooks, and top-1 query retrieval.

## Write Routing

For entity key `q_e` and slot code `S_j`, the router score is cosine similarity:

`a_j = cosine(q_e, S_j)`

Occupied slots are masked. The hard forward slot is the stable argmax among available slots. `kv_first_free` instead chooses the lowest available slot. `kv_fixed_position` uses the binding index.

For `kv_annealed_router`, gradients use a straight-through surrogate:

`g_j = one_hot(j*) + p_j - stop_gradient(p_j)`

where `p = softmax(a / tau)`. The selected slot receives the exact key/value in the forward pass; the surrogate distributes gradients to slot scores. Temperature decreases linearly from 2.0 to 0.1 through update 599. Updates 600–799 use hard routing without a soft surrogate.

The entity token stores its assignment until the immediately following value token, which writes the value to the same slot.

## Dataset

Structural samples retain the Phase 6 entity-value holdout rule. Binding pairs are deterministically permuted per sample, so an entity can occur at any binding position and position cannot encode a slot address. Query target and literal train/holdout isolation remain unchanged.

## Telemetry

Each sequence records:

- entity write slots;
- query read slot;
- collision count;
- hard/soft disagreement count;
- routing entropy sum and decision count;
- final hard-only window collision and entropy metrics;
- per-slot write load.

The CLI records training and holdout routing metrics, exact online sample count, router temperature schedule, and causal deltas.

## Interventions

- `randomized_write_slots`: deterministic cyclic replacement of each chosen write slot.
- `cleared_writer_assignment`: value tokens cannot reuse the entity assignment.
- `allow_write_collisions`: occupancy mask is disabled.
- `permuted_workspace_keys`.
- `zero_query_key`.
- `no_mechanism_output`.

## Formal Gates

For 30 paired seeds, 800 updates, batch size 8:

1. `kv_first_free` holdout accuracy is at least 0.99 in every seed.
2. `kv_annealed_router` mean holdout accuracy is at least 0.95 and significantly exceeds the content-blind Phase 6 control in NLL, Brier, and accuracy.
3. Final 200-update collision rate is below 0.01.
4. Every inference binding writes exactly one slot and every query reads exactly one slot.
5. Every intervention significantly degrades the annealed model; removing KV output must not improve it.
6. `kv_hard_router` is diagnostic only and is not required to pass.

## Boundaries

Success would establish learned collision-free slot addressing under a fixed three-binding topology. It would not establish learned mechanism selection, variable workspace size, variable binding count, or natural-language variable binding.
