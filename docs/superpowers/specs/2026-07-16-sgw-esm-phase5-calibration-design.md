# SGW-ESM Phase 5 Structural Generalization and Calibration Design

## Objective

Test whether the fixed sparse mediation path established in Phase 4 can generalize under a non-repeating online training distribution, and whether bounded logits repair its severe overconfidence without removing workspace causality.

## Structural task

The sequence remains three entity-value bindings followed by a query. For each entity `e`, the pair `(e, e mod value_count)` is reserved for structural holdout. Online training never contains a reserved pair. Holdout samples require the queried binding to use the reserved pair while distractors remain training-compatible.

The deterministic training stream enumerates and shuffles all valid training sequences without replacement. The formal protocol consumes 6,400 unique samples per condition (`800 × 8`).

## Conditions

- `structural_core_full`: full-content recurrent control.
- `structural_core_blind`: content-blind prior control.
- `structural_mediation_linear`: fixed mediation with the Phase 4 linear output.
- `structural_mediation_bounded`: identical parameters and active compute, with `logits = tanh(raw_logits)`.

## Metrics

Primary metrics are holdout NLL, accuracy, multiclass Brier score and 10-bin ECE. Secondary diagnostics are mean maximum confidence and mean true-class probability.

## Positive gate

The bounded mediation condition must:

1. improve NLL, Brier and ECE over linear mediation with paired confidence intervals above zero;
2. retain accuracy and true-class probability;
3. outperform the content-blind control on NLL and accuracy;
4. retain harmful-to-remove workspace-write and reader-inbox interventions.

This phase does not test learned routing or autonomous specialization.
