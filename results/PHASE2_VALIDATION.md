# SGW-ESM Phase 2 Formal Validation

## Status

Phase 2 completed on GitHub Actions run `29437371946` using 30 paired seeds. All six experiment shards, deterministic aggregation, checksum verification, and artifact upload completed successfully. The artifact archive SHA-256 is:

```text
740b650442da338803eb64629d491a1617afa72a0d09a8d75fc5b7ffafb51038
```

The consolidated artifact contains 120 model runs and 150 evaluation-only causal interventions. Independent local reaggregation from the downloaded raw shards produced byte-identical output files and verified every manifest checksum.

## Protocol

- seeds: `0..29`;
- optimization steps: 400;
- batch size: 8;
- unique training sequences: 160;
- unique disjoint holdout sequences: 48;
- target classes: 5;
- paired bootstrap replicates: 50,000;
- bootstrap seed: `20260715`.

Controls:

| Preset | Parameters | Estimated MAdds/token |
|---|---:|---:|
| `core_small` | 327 | 210 |
| `core_compute_matched` | 2,232 | 1,092 |
| `core_param_matched` | 2,181 | 1,474 |
| `sgw` | 2,182 | 1,092 |

The compute-matched core exactly matches SGW's estimated active compute. The parameter-matched core differs by one scalar but uses more estimated compute. A positive SGW claim was prespecified to require both a 95% paired bootstrap interval excluding zero and the same direction in at least 24 of 30 seeds.

## Aggregate performance

| Preset | Final train NLL | Final holdout NLL | Holdout accuracy |
|---|---:|---:|---:|
| `core_small` | 0.3741 ± 0.1302 | **1.5292 ± 0.2698** | 0.4944 ± 0.0599 |
| `core_compute_matched` | 0.3010 ± 0.0931 | 1.6587 ± 0.3365 | 0.5375 ± 0.0857 |
| `core_param_matched` | **0.1072 ± 0.0697** | 1.7545 ± 0.2856 | **0.5542 ± 0.0562** |
| `sgw` | 0.2326 ± 0.0688 | 1.5660 ± 0.2210 | 0.5438 ± 0.0625 |

All larger models strongly fit the training set. Their generalization gaps remain large:

| Preset | Mean holdout NLL minus train NLL |
|---|---:|
| `core_small` | 1.1551 |
| `core_compute_matched` | 1.3577 |
| `core_param_matched` | 1.6473 |
| `sgw` | 1.3335 |

The task is therefore dominated by generalization and representation formation rather than inability to reduce training loss.

## Paired control comparisons

Positive `baseline - SGW` means SGW has lower holdout NLL.

| Comparison | Mean difference | 95% bootstrap CI | Positive / negative seeds | Gate |
|---|---:|---:|---:|---|
| `core_small - SGW` | -0.0369 | [-0.1325, 0.0607] | 11 / 19 | fail |
| `core_compute_matched - SGW` | 0.0926 | [-0.0347, 0.2220] | 19 / 11 | fail |
| `core_param_matched - SGW` | 0.1885 | [0.0671, 0.3109] | 18 / 12 | fail direction gate |

The parameter-matched comparison has a positive mean and bootstrap interval, but the direction reverses in 12 of 30 seeds and misses the prespecified 24/30 stability gate. SGW therefore has not demonstrated a robust advantage over either fair control.

## Causal workspace interventions

Positive NLL increase means removing or corrupting a pathway harmed the trained SGW.

| Intervention | Mean NLL increase | 95% bootstrap CI | Positive / negative seeds | Mean accuracy change |
|---|---:|---:|---:|---:|
| `no_broadcast` | **-0.0205** | **[-0.0403, -0.0009]** | 10 / 20 | -0.0125 |
| `no_workspace_persistence` | 0.0342 | [-0.0207, 0.0899] | 17 / 13 | -0.0389 |
| `no_workspace_output` | -0.0115 | [-0.0368, 0.0131] | 14 / 16 | -0.0160 |
| `permuted_recipients` | -0.0103 | [-0.0367, 0.0162] | 12 / 18 | -0.0083 |

No workspace pathway satisfies the intended causal-support criterion. The only interval excluding zero points in the opposite direction: disabling mechanism broadcast slightly improves NLL. Accuracy falls slightly, so the broadcast path may affect confidence and class decisions differently, but it is not a demonstrated beneficial modeling pathway.

## Routing diagnostics

Every SGW run obeyed exact budgets of two active mechanisms, one workspace writer, and two recipients per token. The mean mechanism activation fractions were approximately:

```text
[0.1624, 0.1586, 0.1845, 0.1781, 0.1602, 0.1562]
```

Normalized activation entropy across seeds was `0.9489 ± 0.0292`, where 1.0 is perfectly uniform. Aggregate load is therefore close to uniform. This is not evidence of semantic specialization; no role-conditioned or intervention-based specialization has yet been established.

## Scientific conclusion

Phase 2 rejects a stronger interpretation of the current architecture:

1. SGW is trainable and its hard sparse budgets are exact.
2. It does not robustly outperform parameter- or compute-matched recurrent controls.
3. The workspace is not causally necessary under the current task and loss.
4. Recipient-selected mechanism broadcast is slightly harmful in NLL under the registered intervention.
5. Router load remains near uniform and does not establish mechanism specialization.

The current architecture contains bypass paths: the recurrent spine reads pooled workspace directly and the classifier reads workspace directly, while mechanisms also receive recipient-selected inbox broadcasts. The final loss can therefore bypass the intended mechanism-to-workspace-to-broadcast coordination loop. Phase 3 should isolate these paths before increasing model size.

## Phase 3 gate

The next experiment should compare:

- the current redundant-read SGW;
- a broadcast-forced SGW with no direct workspace-to-spine or workspace-to-output read;
- final-loss-only training;
- aligned workspace auxiliary supervision with coefficient annealing;
- parameter- and compute-matched recurrent controls.

It should add `no_spine_workspace`, `no_mechanism_output`, and complete workspace-disconnect interventions, role-conditioned routing matrices, and per-mechanism causal ablations. A successful gate requires both matched-control improvement and degradation under removal of the specific pathway claimed to provide the gain.
