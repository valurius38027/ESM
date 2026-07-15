# SGW-ESM Phase 4 Formal Validation

## Status

Phase 4 is complete for 30 paired seeds (`0..29`). The fixed sparse workspace demonstrates a real information-mediation capacity, but the full modeling gate fails under proper log-loss evaluation.

The defensible conclusion is:

> A content-separated fixed writer-slot-reader workspace can carry binding information that is causally necessary for above-chance classification. The current learner nevertheless overfits and is severely miscalibrated, so this is not yet evidence of a strong or efficient general modeling mechanism.

Phase 4 is a capacity diagnostic. It does not claim learned routing, autonomous specialization, or superiority over recurrent controls.

## Protocol and integrity

- 30 paired seeds;
- 400 optimization steps;
- batch size 8;
- 160 unique training sequences;
- 48 unique, disjoint holdout sequences;
- 50,000 paired bootstrap replicates;
- deterministic analysis seed `20260715`;
- six five-seed GitHub Actions shards;
- formal workflow run `29445715336`;
- four-configuration CI run `29445715258`;
- Actions artifact digest `sha256:7007a370b7b44d7a8c8f2cdd75d5987029daa7d70782ac8047b4cd39e982e081`.

The artifact contains 120 condition runs and 300 evaluation-only causal interventions. All eight consolidated files pass the committed `SHA256SUMS`. Independent local reaggregation from the downloaded raw shard files was byte-identical to the Actions output.

## Architecture boundary

The mediation preset removes the Phase 3 information bypasses:

- the recurrent spine receives no content embeddings;
- the spine receives no workspace input;
- the classifier reads neither spine nor workspace directly;
- three fixed writer mechanisms receive the three binding pairs;
- each writer uses one fixed workspace slot;
- only the query reader receives broadcast;
- the final classifier reads only the final reader mechanism state;
- learned router, writer, slot and recipient parameters are not allocated.

The topology is therefore a test of sparse workspace mediation capacity, not a learned-routing experiment.

## Main results

Chance NLL for five target values is `ln(5) = 1.609438` and chance accuracy is `0.20`.

| Condition | Parameters | Estimated MAdds/token | Holdout NLL | Holdout accuracy |
|---|---:|---:|---:|---:|
| `core_full_content` | 1,169 | 984 | 2.1878 ± 0.3565 | 0.4424 ± 0.0827 |
| `core_content_blind` | 725 | 696 | 1.6208 ± 0.0232 | 0.2146 ± 0.0638 |
| `mediation_final_only` | 15,650 | 6,784 | 1.8973 ± 0.2911 | 0.4507 ± 0.0850 |
| `mediation_aux_annealed` | 15,650 | 6,784 | 2.0555 ± 0.2790 | 0.4319 ± 0.0766 |

### Mediation-capacity result: positive for classification information

`mediation_final_only` exceeds the content-blind control in holdout accuracy by `0.23611` on average. The paired bootstrap CI is approximately `[0.20000, 0.27083]`, and all 30 seeds favor the mediation model.

The auxiliary mediation model exceeds the content-blind control in accuracy by `0.21736`, with CI approximately `[0.18611, 0.24792]`; all 30 seeds again favor mediation.

This establishes that the fixed workspace path transmits binding information unavailable to the blind recurrent control.

### Proper-scoring modeling gate: failed

The same models remain worse than the content-blind prior under NLL:

- `mediation_final_only` is `0.28785` NLL above chance on average;
- `mediation_aux_annealed` is `0.44604` above chance;
- the content-blind control is only `0.01141` above chance.

The fixed mediation model obtains useful argmax decisions but assigns excessive probability to wrong answers. It is therefore informative but badly calibrated.

The full-content recurrent control shows the same broader failure mode: training NLL falls to `0.0833`, while holdout NLL rises to `2.1878`. Its mean train-to-holdout NLL gap is `2.1045`. The mediation-final gap is `1.5043`, and the auxiliary mediation gap is `1.7268`.

### Comparison with full-content core

`mediation_final_only` improves NLL over `core_full_content` by `0.29055`, with 95% CI `[0.15705, 0.42238]`, while holdout accuracy is statistically indistinguishable (`+0.00833`, CI approximately `[-0.02639, 0.04028]`).

Thus fixed sparse mediation generalizes better than the small dense recurrent control on this limited corpus, but neither is well calibrated and neither beats the content-blind prior on NLL.

### Auxiliary-supervision gate: failed

For `mediation_final_only` minus `mediation_aux_annealed`:

- mean NLL difference: `-0.15819`;
- 95% CI: `[-0.25474, -0.06169]`;
- only 10/30 seeds favor auxiliary training.

The annealed auxiliary objective significantly worsens final holdout NLL. Accuracy changes by `-0.01875`, with a CI crossing zero.

This is not caused by incomplete withdrawal:

- final auxiliary coefficient is exactly `0.0` in all 30 seeds;
- final weighted auxiliary contribution is exactly `0.0` in all 30 seeds;
- primary loss decreases by `0.34604` on average from the anneal-boundary window to the final window, with CI approximately `[0.27370, 0.40658]`.

Temporary direct workspace supervision therefore changes the optimization path in an unfavorable way rather than merely remaining active at evaluation.

## Causal mediation gates

The following interventions are applied to the trained auxiliary mediation model without retraining.

| Intervention | Mean NLL increase | 95% bootstrap CI | Mean accuracy change |
|---|---:|---:|---:|
| `no_workspace_writes` | 0.48169 | [0.32893, 0.63713] | -0.21667 |
| `no_broadcast` | 0.27292 | [0.13507, 0.41907] | -0.21319 |
| `zero_reader_inbox` | 0.27292 | [0.13436, 0.41790] | -0.21319 |
| `no_workspace_persistence` | 0.19771 | [0.06419, 0.33441] | -0.13542 |
| `permuted_recipients` | 0.27292 | [0.13433, 0.41655] | -0.21319 |

Accuracy falls under `no_broadcast`, `no_workspace_writes` and `zero_reader_inbox` in all 30 seeds. Approximate paired accuracy CIs are:

- `no_workspace_writes`: `[-0.25208, -0.18125]`;
- `no_broadcast`: `[-0.24653, -0.17986]`;
- `zero_reader_inbox`: `[-0.24722, -0.17917]`.

These are direct positive causal results: correct workspace writes and delivery to the query reader are necessary for the learned above-chance classifier.

`no_workspace_output` and `no_spine_workspace` are exact zero-effect controls, as expected, because those paths are structurally absent in the mediation preset.

### Output-calibration diagnostic

`no_mechanism_output` improves NLL by `0.43960` on average, with CI `[-0.53473, -0.33742]` when expressed as intervention NLL increase, but lowers accuracy by `0.24722`.

This apparent contradiction is diagnostic rather than beneficial: removing the only informative output path returns predictions toward an uninformative prior, which improves log loss relative to an overconfident wrong model while destroying classification information.

## Gate decision

Phase 4 separates two claims that were conflated in earlier phases:

1. **Can a sparse workspace carry task-relevant binding information through a unique cross-mechanism path?** Yes, under a fixed topology.
2. **Does the current training system convert that information into a well-generalized, calibrated predictor?** No.

The phase therefore partially passes the mediation-capacity gate and fails the modeling-quality gate.

## Next phase

Do not restore learned routing yet. Phase 5 should retain the fixed information-separated topology and target the observed generalization/calibration failure:

1. replace the repeatedly sampled 160-example corpus with a deterministic streaming training distribution;
2. use a structural entity-value holdout rather than literal-sequence disjointness alone;
3. add bounded or normalized reader-to-class logits;
4. measure Brier score, expected calibration error, maximum confidence and true-class probability in addition to NLL and accuracy;
5. retain `no_workspace_writes` and `zero_reader_inbox` causal gates;
6. require the intact mediation model to beat the content-blind prior on NLL before any return to autonomous routing.
