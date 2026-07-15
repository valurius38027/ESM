# SGW-ESM Phase 3 Credit Assignment and Pathway Isolation Design

## Objective

Determine whether the Phase 2 failure is caused by architectural bypass paths and insufficient credit assignment rather than insufficient representational capacity.

Phase 3 must answer two separate questions:

1. Can a sparse workspace-to-broadcast-to-mechanism pathway be trained when direct workspace bypasses are removed?
2. Does aligned, temporary workspace supervision establish a useful pathway that remains functional after the auxiliary coefficient reaches zero?

## Scope

Phase 3 remains a deterministic, CPU-only controlled experiment. It does not add external episodic memory, dynamic depth, online parameter updates, natural-language corpora, or a production inference kernel.

## Architecture conditions

### `sgw_redundant`

The Phase 2 SGW architecture:

- the recurrent spine reads pooled previous workspace;
- mechanisms read recipient-selected inboxes;
- the final classifier reads spine, workspace, and active-mechanism summary.

This condition retains the direct bypass paths.

### `sgw_broadcast_forced`

The same stored parameter set and sparse budgets, but primary prediction disables:

- direct workspace-to-spine input;
- direct workspace-to-output projection.

The workspace can affect primary prediction only through recipient-selected inbox broadcast, later mechanism activation, and the active-mechanism output summary. The recurrent spine still reads token embeddings and its own recurrent state, so the matched recurrent controls remain valid.

The workspace output projection remains allocated solely as an auxiliary training readout and is excluded from primary inference.

## Training conditions

Phase 3 compares:

- `core_small`;
- `core_compute_matched`;
- `core_param_matched`;
- `sgw_redundant_final_only`;
- `sgw_broadcast_forced_final_only`;
- `sgw_broadcast_forced_aux_annealed`.

For the auxiliary condition, the final workspace state predicts the same downstream target using the existing workspace projection. Total loss at step `t` is:

```text
L(t) = L_primary + lambda(t) * L_workspace
```

with:

```text
lambda(t) = lambda_0 * max(0, 1 - t / T_anneal)
lambda_0 = 0.5
T_anneal = 200 steps
```

The formal run lasts 400 steps, so the final 200 steps use primary loss only. Evaluation never includes auxiliary loss or the auxiliary workspace readout.

This is an aligned diagnostic intervention, not evidence of autonomous workspace formation. Its purpose is to distinguish trainability/credit-assignment failure from architectural incapacity.

## Evaluation-only interventions

All SGW conditions support:

- `intact`;
- `no_broadcast`;
- `no_workspace_persistence`;
- `permuted_recipients`;
- `no_spine_workspace`;
- `no_workspace_output`;
- `no_mechanism_output`;
- `workspace_disconnected`.

`workspace_disconnected` simultaneously disables workspace-to-spine input, workspace-to-output projection, and mechanism inbox broadcast. It does not delete mechanism recurrence or direct token input.

Interventions never mutate parameters.

## Routing diagnostics

Every evaluation reports activation counts conditioned on token role:

- `entity`;
- `value`;
- `filler`;
- `query_marker`;
- `query_entity`.

The output is a deterministic role-by-mechanism count matrix. Aggregate specialization is not inferred from marginal load alone. Phase 3 reports:

- normalized marginal activation entropy;
- normalized conditional entropy `H(Mechanism | Role)`;
- mutual information `I(Role; Mechanism)`;
- role-wise dominant mechanism stability across seeds.

These statistics describe routing dependence but do not by themselves prove semantic function. Causal per-mechanism ablation remains required for a positive specialization claim.

## Formal protocol

- paired seeds: `0..29`;
- optimization steps: 400;
- batch size: 8;
- train samples: 160 unique sequences;
- holdout samples: 48 unique disjoint sequences;
- bootstrap replicates: 50,000;
- fixed analysis seed: `20260715`.

GitHub Actions executes six five-seed shards and publishes raw and consolidated artifacts.

## Prespecified gates

### Credit-assignment gate

`sgw_broadcast_forced_aux_annealed` must improve holdout NLL over `sgw_broadcast_forced_final_only` with:

- paired 95% bootstrap interval excluding zero;
- improvement in at least 24 of 30 seeds.

### Matched-control gate

The auxiliary broadcast-forced model must improve over both matched recurrent controls using the same two criteria. Stored parameters and active inference compute are reported separately.

### Broadcast causal gate

For the auxiliary broadcast-forced model, both `no_broadcast` and `permuted_recipients` must worsen holdout NLL with paired 95% intervals excluding zero. `workspace_disconnected` must cause at least as much degradation as either individual intervention.

### Auxiliary withdrawal gate

The workspace coefficient is exactly zero for the final 200 optimization steps. The reported final primary loss must remain below its value at the anneal boundary. This prevents claiming success from a permanently active auxiliary head.

### Specialization gate

A positive specialization claim requires:

- role-routing mutual information above zero with a bootstrap interval excluding zero;
- a stable role-to-mechanism pattern in at least 24 of 30 seeds;
- measurable degradation from ablating the implicated mechanism on the corresponding role-sensitive examples.

Phase 3 initially implements routing statistics and mechanism ablation data; it must not claim specialization unless all three conditions pass.

## Scientific interpretation

Possible outcomes are deliberately separated:

- Auxiliary succeeds and broadcast interventions hurt: current failure is primarily credit assignment.
- Auxiliary improves workspace encoding but broadcast interventions do not hurt: the primary path still bypasses or ignores broadcast.
- Forced architecture fails even with auxiliary supervision: current broadcast topology or state dynamics are inadequate.
- Final-only forced architecture succeeds: Phase 2 failure was primarily bypass competition rather than gradient weakness.
- None beats matched cores: workspace coordination remains an unnecessary inductive bias for this task.
