# Phase 5 Structural Calibration Validation

## Scope

Phase 5 retains the fixed sparse mediation topology established in Phase 4 and tests two narrower hypotheses:

1. repeated training on a finite corpus caused memorization rather than a reusable binding procedure;
2. unbounded linear logits caused the severe probability overconfidence observed in Phase 4.

The phase therefore adds a deterministic no-replacement training stream, an entity-value structural holdout, a parameter-free bounded logit head, and explicit calibration metrics. It does **not** test learned routing or autonomous specialization.

## Pre-registered conditions

| Condition | Content path | Output head |
|---|---|---|
| `structural_core_full` | recurrent core sees full content | linear |
| `structural_core_blind` | recurrent core is content blind | linear |
| `structural_mediation_linear` | fixed writer-slot-reader mediation | linear |
| `structural_mediation_bounded` | fixed writer-slot-reader mediation | `tanh`-bounded to `[-1, 1]` |

All conditions consume exactly `800 × 8 = 6,400` deterministic online training samples per seed. The structural stream does not repeat a sample before exhaustion, and the queried entity-value combinations in holdout are excluded from training.

Chance values for the five-way target are:

- NLL: `ln(5) = 1.609437912434`;
- accuracy: `0.2`;
- multiclass Brier score for a uniform predictor: `0.8`.

## Execution and verification

Repository: `valurius38027/ESM`

Branch: `agent/phase5-calibration`

Draft PR: `#4`

CI run: `29470148046`

Formal experiment run: `29470148085`

Formal artifact:

- artifact ID: `8364540867`;
- artifact name: `phase5-formal-results`;
- ZIP SHA-256: `104adf7dd4e520e464876bd8ab72c1320a89197245fc74eca6ca1b8e164b1825`.

The formal run used:

- seeds `0..29`;
- 800 optimizer steps per condition;
- batch size 8;
- 160 diagnostic training samples;
- 160 structural holdout samples;
- six parallel seed shards;
- 50,000 paired bootstrap replicates;
- deterministic analysis seed `20260716`.

Validation evidence:

- 58 focused C++ tests;
- deterministic CLI test;
- deterministic Phase 2, 3, 4, and 5 aggregation tests;
- GCC Debug, Clang Debug, GCC Release, and ASan/UBSan all passed;
- 120 formal condition runs and 300 causal evaluations;
- all formal files pass the committed `SHA256SUMS`;
- downloaded ZIP SHA equals the GitHub artifact digest;
- independently parsed raw shards reproduce all point estimates exactly;
- independently generated 200,000-replicate vectorized paired bootstrap intervals corroborate the reported signs and interval separation;
- the independently rerun aggregator produced byte-identical `phase5_runs.csv`, `phase5_causal.csv`, `phase5_summary.csv`, and `phase5_paired.csv` before the local command ceiling interrupted the remaining pure-Python bootstrap tables.

## Formal results

### Mean holdout metrics across 30 seeds

| Condition | NLL | Accuracy | Brier | ECE | Max confidence |
|---|---:|---:|---:|---:|---:|
| `structural_core_blind` | **1.647312** | **0.173750** | **0.814091** | **0.054082** | 0.227784 |
| `structural_core_full` | 3.207506 | 0.030208 | 1.419658 | 0.673720 | 0.701264 |
| `structural_mediation_linear` | 3.396669 | 0.045000 | 1.423301 | 0.666376 | 0.708306 |
| `structural_mediation_bounded` | 2.723207 | 0.009583 | 1.165377 | 0.398036 | 0.405522 |

The content-blind control remains closest to the uniform prior. Every content-using model strongly overfits the training distribution and fails on the reserved entity-value combinations.

### Training-to-holdout gap

| Condition | Train NLL | Holdout NLL | Train accuracy | Holdout accuracy |
|---|---:|---:|---:|---:|
| `structural_core_blind` | 1.608682 | 1.647312 | 0.208125 | 0.173750 |
| `structural_core_full` | 0.833256 | 3.207506 | 0.578333 | 0.030208 |
| `structural_mediation_linear` | 0.823407 | 3.396669 | 0.567708 | 0.045000 |
| `structural_mediation_bounded` | 1.188591 | 2.723207 | 0.459375 | 0.009583 |

The online stream removes exact sample repetition, but it does not by itself induce a compositional binding algorithm. The models learn training-distribution correlations that reverse under the structural holdout.

## Bounded versus linear mediation

The bounded head succeeds at its narrow calibration objective:

| Metric | Mean improvement | 95% paired bootstrap CI | Favorable seeds |
|---|---:|---:|---:|
| NLL | 0.673462 | [0.539355, 0.800561] | 28/30 |
| Brier | 0.257924 | [0.231460, 0.283241] | 30/30 |
| ECE | 0.268340 | [0.244480, 0.292730] | 30/30 |

However, bounded mediation reduces accuracy by `0.035417`, with CI `[-0.045625, -0.025625]`. It also remains substantially worse than the content-blind control:

- NLL deficit: `1.075895`, 0/30 seeds favorable;
- Brier deficit: `0.351286`, 0/30 seeds favorable;
- ECE deficit: `0.343954`, 0/30 seeds favorable;
- accuracy deficit: `0.164167`, 0/30 seeds favorable.

Therefore logit bounding mitigates overconfidence but does not repair the underlying representation or algorithm.

## Causal mediation diagnostics

The bounded model remains causally dependent on the fixed workspace path:

- removing broadcast raises NLL by `0.344549`, CI `[0.278868, 0.411832]`;
- removing workspace writes raises NLL by `0.331851`, CI `[0.271041, 0.394650]`;
- zeroing the reader inbox raises NLL by `0.344549`, CI `[0.277937, 0.411366]`;
- removing workspace persistence raises NLL by `0.295378`, CI `[0.243427, 0.347830]`.

These interventions reduce holdout accuracy by approximately `0.009583`, which is the entire already-small intact accuracy. Thus the workspace path is causally active, but its learned content is not structurally correct.

The strongest diagnostic is `no_mechanism_output`:

- NLL improves by `1.094514`;
- accuracy improves by `0.156875`;
- Brier improves by `0.357642`;
- ECE improves by `0.333953`.

After removing the mechanism output, the model returns close to the content-blind prior. This shows that the fixed mediation branch carries a systematic **anti-generalizing residual**, not merely random noise.

`no_workspace_output` and `no_spine_workspace` have exactly zero effect, as expected for the Phase 4/5 mediation preset: those paths are not used by the classifier.

## Scientific verdict

Phase 5 partially passes only the calibration sub-gate:

> Bounding logits is an effective parameter-free method for reducing the severity of overconfidence in the fixed mediation model.

It fails the modeling gate:

> Neither a no-replacement online stream nor bounded logits produces structural binding generalization. The fixed workspace path remains causally active but learns a systematically wrong mapping on held-out entity-value combinations.

This result narrows the next problem. The primary obstacle is no longer corpus repetition or logit scale. It is the absence of an architectural or objective-level constraint that forces value transport to remain identity-preserving and query-conditioned across unseen bindings.

## Required next gate

Phase 6 should not return to learned routing yet. It should retain the fixed sparse topology and compare:

1. the current unconstrained writer/reader transforms;
2. an explicit identity-preserving value transport path;
3. a contrastive binding objective that separates queried from distractor slots;
4. a permutation-equivariant or key-value factorized representation.

A positive Phase 6 result must beat the content-blind control on structural holdout **and** retain harmful workspace-path interventions without relying on output overconfidence.
