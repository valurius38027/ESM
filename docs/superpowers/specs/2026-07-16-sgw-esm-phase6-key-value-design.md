# SGW-ESM Phase 6 Identity-Preserving Key-Value Mediation Design

## Objective

Determine whether the Phase 5 structural anti-generalization is caused by losing slot identity and value identity inside the fixed sparse workspace. Phase 6 replaces pooled free-form mediation with explicit sparse key-value storage and query-conditioned retrieval while keeping the structural holdout, no-replacement training stream, CPU-only runtime, and causal intervention discipline unchanged.

Phase 6 is an architectural-inductive-bias gate. It does not claim autonomous learned routing: write slots remain fixed by binding position and the reader performs one sparse top-1 lookup.

## Conditions

The formal paired experiment uses four conditions on the same 30 seeds and structural split:

1. `structural_core_blind`: the Phase 5 content-blind recurrent control.
2. `structural_mediation_bounded`: the Phase 5 bounded free-form mediation control.
3. `structural_kv_exact`: a parameter-free symbolic capacity gate.
4. `structural_kv_learned`: a tied, trainable key-value codebook with normalized sparse retrieval.

All conditions consume 800 updates at batch size 8 from the deterministic no-replacement structural stream. The exact condition consumes the stream for paired accounting even though it has no trainable parameters.

## Workspace Representation

For each of three binding slots, the workspace stores two disjoint fields:

- a key field representing the entity identity;
- a value field representing the value-class identity.

The workspace dimension is `key_dim + value_dim`. Entity tokens write only the key field. The following value token writes only the value field and preserves the key. No learned recurrent write transform, gate, pooled broadcast, or cross-slot mixing is permitted in either Phase 6 condition.

## 6A: Exact Capacity Gate

`structural_kv_exact` stores the literal entity token and value class for each slot. At the final query entity token, the reader selects the slot whose stored entity token is equal to the query token. It emits a fixed positive logit for the stored value class and zero for other classes.

This model has zero trainable scalars. It must achieve at least 99% holdout accuracy and holdout NLL below 0.05 for every seed. Failure means the task wiring, structural split, or intervention implementation is incorrect and blocks 6B interpretation.

## 6B: Learned Tied Key-Value Gate

`structural_kv_learned` owns two parameter matrices:

- `kv_entity_codebook[entity_count, key_dim]`;
- `kv_value_codebook[value_count, value_dim]`.

The same normalized entity-code row is used when writing a slot key and when encoding the final query. This identity-preserving tie makes matching permutation-equivariant over entity labels. The reader computes cosine similarity to each occupied slot and selects exactly one slot using deterministic stable top-1 routing.

The selected normalized value code is decoded with the same normalized `kv_value_codebook` rows:

`logit_c = logit_scale * cosine(retrieved_value, kv_value_codebook[c])`.

The codebooks remain trainable; the shared encoder/decoder tie is the architectural constraint under test. No output bias, untied classifier, recurrent reader transform, or direct token-to-output bypass is allowed.

## Interventions

Phase 6 adds:

- `permuted_workspace_keys`: cyclically permute slot keys while leaving values in place;
- `zero_query_key`: replace the final query key with zero before top-1 selection.

The existing interventions are mapped as follows:

- `no_workspace_writes`: preserve routes but suppress key and value writes;
- `no_workspace_persistence`: clear slots before every token;
- `no_broadcast`, `zero_reader_inbox`, `workspace_disconnected`, `no_workspace_output`, and `no_mechanism_output`: suppress the retrieved value contribution;
- `permuted_recipients`: remains a legacy intervention and is not part of the Phase 6 formal comparison.

Route traces record the fixed active mechanism, intended writer and writer slot, reader recipient, and selected read slot. Interventions must not alter the intended write-route trace.

## Formal Gates

### Capacity gate

For `structural_kv_exact`, every seed must satisfy:

- holdout accuracy >= 0.99;
- holdout NLL < 0.05.

### Learned structural gate

For `structural_kv_learned` versus `structural_core_blind`:

- paired mean holdout NLL improvement is positive with a 95% bootstrap interval excluding zero;
- paired mean Brier improvement is positive with a 95% bootstrap interval excluding zero;
- paired mean accuracy improvement is positive with a 95% bootstrap interval excluding zero;
- at least 28 of 30 seeds improve on each of those three metrics;
- mean holdout accuracy is at least 0.95.

The learned model must also outperform `structural_mediation_bounded` on NLL and accuracy with intervals excluding zero.

### Causal gate

For `structural_kv_learned`, each of `no_workspace_writes`, `no_workspace_persistence`, `permuted_workspace_keys`, `zero_query_key`, and `no_mechanism_output` must worsen NLL and accuracy relative to intact operation. Removing the mediation output must not improve either metric.

## Determinism and Validation

- C++20, standard library only, CPU only.
- Existing Phase 1-5 behavior and result parsers remain unchanged.
- GCC Debug, Clang Debug, GCC Release, and ASan/UBSan must pass.
- The formal matrix uses six shards, 30 paired seeds, 800 steps, batch size 8, and 50,000 deterministic paired bootstrap replicates.
- Aggregation rejects missing, duplicated, extra, or mismatched files and verifies exact sample consumption.
- Consolidated results include SHA-256 checksums and a bounded scientific validation report.
