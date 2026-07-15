# SGW-ESM Phase 2 Experimental Controls Design

## Objective

Determine whether the Phase 1 sparse global workspace provides modeling value beyond a recurrent core under fair parameter and active-compute budgets, and whether its broadcast and persistent workspace pathways are causally used after training.

## Scope

Phase 2 remains a deterministic CPU-only toy research system. It does not add external memory, dynamic depth, online structural growth, natural-language data, or a high-performance kernel.

## Baselines

All conditions use the same delayed-binding task, split, optimizer, training schedule, seed, and evaluation samples.

- `core_small`: Phase 1 recurrent baseline (`embedding_dim=8`, `spine_dim=8`).
- `core_compute_matched`: recurrent baseline with `embedding_dim=66`, `spine_dim=13`.
- `core_param_matched`: recurrent baseline with `embedding_dim=40`, `spine_dim=22`.
- `sgw`: Phase 1 sparse global workspace configuration.

For the fixed task vocabulary and output size:

- `core_compute_matched`: 2,232 parameters and 1,092 estimated multiply-adds/token.
- `core_param_matched`: 2,181 parameters and 1,474 estimated multiply-adds/token.
- `sgw`: 2,182 parameters and 1,092 estimated multiply-adds/token.

No integer embedding/spine pair matches both constraints exactly. The two controls separately match active compute exactly and parameter count to one scalar; claims must survive both comparisons. `core_small` remains as a low-cost reference.

## Causal interventions

The same trained SGW instance is evaluated under five deterministic variants:

1. `intact`: normal execution.
2. `no_broadcast`: recipient selection is retained for trace comparability, but recipient inboxes are not mutated.
3. `no_workspace_persistence`: workspace is zeroed before every token, so current-token writes remain available but cannot persist across token boundaries.
4. `no_workspace_output`: the final classifier cannot directly read workspace; recurrent and mechanism pathways remain intact.
5. `permuted_recipients`: selected recipient IDs are cyclically shifted by one mechanism before broadcast, preserving recipient count and message magnitude while breaking learned addressing.

Interventions are evaluation-only and never alter trained parameters.

## Measurement

Per condition and seed record:

- initial and final train/holdout NLL;
- final holdout accuracy;
- parameter count;
- estimated multiply-adds/token;
- exact mean active mechanisms, writers, and recipients;
- mechanism load vector;
- elapsed wall-clock seconds.

For causal variants record holdout NLL and accuracy relative to intact SGW.

## Formal experiment

Use 30 paired seeds (`0..29`), 400 training steps, batch size 8, 160 unique training samples, and 48 unique holdout samples. GitHub Actions runs six independent five-seed shards and aggregates artifacts deterministically.

Formal inference rules:

- A positive matched-baseline claim requires a paired mean NLL improvement whose 95% bootstrap confidence interval excludes zero and whose direction is stable in at least 24/30 seeds.
- A pathway is causally supported only if its intervention worsens NLL with a 95% paired bootstrap interval excluding zero.
- Accuracy is secondary because the toy task produces coarse, discrete outcomes.
- No efficiency claim is permitted from estimated MAdds alone.

## Reproducibility

- All routing tie breaks remain stable by index.
- Every CSV uses LF line endings and deterministic ordering.
- Aggregation uses fixed bootstrap seed `20260715` and 50,000 paired bootstrap replicates.
- CI publishes raw shards, consolidated CSVs, a JSON manifest, and SHA-256 hashes.
