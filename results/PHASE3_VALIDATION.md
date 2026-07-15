# SGW-ESM Phase 3 Formal Validation

## Status

Phase 3 is complete for 30 paired seeds (`0..29`). The implementation and formal artifact are reproducible, but the prespecified credit-assignment, matched-control, broadcast-causality, and specialization gates do not pass.

The result does not support the claim that temporary workspace supervision creates a useful global-workspace pathway in the current architecture.

## Protocol and integrity

- 30 paired seeds;
- 400 optimization steps;
- batch size 8;
- 160 unique training sequences;
- 48 unique, disjoint holdout sequences;
- 50,000 paired bootstrap replicates;
- analysis seed `20260715`;
- six five-seed GitHub Actions shards;
- formal workflow run `29442386863`;
- CI workflow run `29442386772`;
- Actions artifact digest `sha256:a8914dca32fe569c4ac06dd5ba9a7fcd061065e6325a5942d6833064327f0b75`.

The artifact contains 180 model runs and 240 evaluation-only causal interventions. All eight consolidated files pass the committed `SHA256SUMS`. Independent local reaggregation from the downloaded raw shard files was byte-identical to the Actions output.

## Main results

| Condition | Parameters | Estimated MAdds/token | Holdout NLL | Holdout accuracy |
|---|---:|---:|---:|---:|
| `core_small` | 327 | 210 | 1.5292 ± 0.2698 | 0.4944 ± 0.0599 |
| `core_compute_matched` | 2,232 | 1,092 | 1.6587 ± 0.3365 | 0.5375 ± 0.0857 |
| `core_param_matched` | 2,181 | 1,474 | 1.7545 ± 0.2856 | 0.5542 ± 0.0562 |
| `sgw_redundant_final_only` | 2,182 | 1,092 | 1.5660 ± 0.2210 | 0.5438 ± 0.0625 |
| `sgw_broadcast_forced_final_only` | 2,182 | 1,002 | 1.5964 ± 0.3279 | 0.5236 ± 0.0847 |
| `sgw_broadcast_forced_aux_annealed` | 2,182 | 1,002 | 1.6882 ± 0.3923 | 0.5271 ± 0.0886 |

### Credit-assignment gate: failed

For `forced_aux` versus `forced_final`, the paired quantity is baseline NLL minus candidate NLL:

- mean: `-0.091774`;
- 95% bootstrap CI: `[-0.200255, 0.018218]`;
- auxiliary condition better in 12/30 seeds and worse in 18/30.

The annealed auxiliary condition is worse on average and meets neither the confidence-interval nor 24/30 directional requirement. Training NLL is effectively unchanged, so the holdout degradation is consistent with failure to improve generalization rather than simple under-optimization.

### Matched-control gate: failed

Against `core_compute_matched`, `forced_aux` is worse by `0.029496` NLL on average, with CI `[-0.194030, 0.129087]` for baseline minus candidate and a 15/15 directional split.

Against `core_param_matched`, `forced_aux` is better by `0.066336` NLL on average, but the CI `[-0.085848, 0.215605]` includes zero and only 17/30 seeds favor SGW. Both matched controls were required.

### Broadcast causal gate: failed

| Intervention | Mean NLL increase | 95% bootstrap CI |
|---|---:|---:|
| `no_broadcast` | 0.008516 | [-0.011012, 0.027646] |
| `no_workspace_persistence` | 0.001924 | [-0.030895, 0.032857] |
| `workspace_disconnected` | 0.008516 | [-0.010988, 0.027764] |
| `permuted_recipients` | 0.012441 | [-0.020848, 0.045352] |

None provides evidence that the learned predictor depends on correct workspace broadcast. `workspace_disconnected` equals `no_broadcast` because the forced preset already disables direct workspace-to-spine and workspace-to-output paths.

### Mechanism-output diagnostic

`no_mechanism_output` changes holdout NLL by `-0.398773`, with CI `[-0.486470, -0.314184]`; all 30 seeds improve in NLL. Holdout accuracy, however, falls by about `0.03681` (post-hoc paired bootstrap CI approximately `[-0.05625, -0.01736]`).

Thus the active-mechanism output branch changes the argmax in some useful cases but is strongly harmful under the proper log-loss scoring rule. The most likely interpretation is poorly calibrated or overconfident residual fusion, not stable workspace-mediated information gain.

### Auxiliary withdrawal gate: only partially instrumented

The schedule implementation is unit-tested to produce weights `0.5` at step 0, `0.0025` at step 199, and exactly `0.0` at steps 200 through 399. The final total-loss and primary-loss windows coincide, and the final weighted auxiliary contribution is zero.

The formal CSV did not record the primary-loss window exactly at the anneal boundary, so the additional boundary-to-final comparison specified in the design cannot be independently reconstructed from the artifact. No positive claim relies on this gate because the earlier required gates already fail. Future schemas must record this boundary explicitly.

### Routing and specialization gate: failed

For the auxiliary condition:

- normalized marginal mechanism entropy: `0.9321 ± 0.0289`;
- role-mechanism mutual information: `0.03364 ± 0.01435` nats;
- normalized mutual information: `0.02090 ± 0.00891`;
- raw-ID dominant-mechanism stability by role: `0.20..0.333`.

Routing is not role-independent, but the dependence is weak and the auxiliary condition has lower mutual information than forced final-only training. Raw mechanism IDs are permutation-symmetric across seeds, so unaligned ID stability is a conservative and imperfect test. More importantly, Phase 3 does not establish role-specific causal function through aligned per-mechanism ablation. No specialization claim is warranted.

## Structural interpretation

The preset called `sgw_broadcast_forced` removes direct workspace reads from the spine and output head, but the recurrent spine still observes the complete token sequence and is read directly by the final classifier. It is therefore an end-to-end information bypass. The experiment forces workspace information to travel through broadcast if the workspace is used, but it does not force the task solution itself to use the workspace.

Phase 3 consequently tests whether an optional workspace residual can become useful under temporary supervision. It does not yet test whether a global workspace can mediate information that is unavailable through any local channel.

## Decision

Do not scale this topology. Phase 4 should use an information-separated mediation task and architecture:

1. the always-on spine receives control/role information but not binding content;
2. writer and reader mechanisms receive disjoint content views;
3. the query reader cannot recover the answer from its local recurrent state;
4. workspace broadcast is the only permitted cross-mechanism content path;
5. output fusion is gated or normalized rather than an unconstrained additive mechanism residual;
6. routing specialization is evaluated after permutation-invariant mechanism alignment and role-conditional causal ablation;
7. the exact anneal-boundary primary loss is persisted in every run row.

This next phase separates architectural mediation capacity from autonomous routing and from output calibration.
