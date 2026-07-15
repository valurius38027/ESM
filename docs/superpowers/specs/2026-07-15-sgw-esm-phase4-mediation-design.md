# SGW-ESM Phase 4 Information-Separated Mediation Design

## Objective

Determine whether a sparse workspace can carry binding content when no recurrent or output bypass can solve the task directly. Phase 4 is a capacity and mediation gate, not an autonomous-routing claim.

## Task

Each sequence contains three entity-value bindings followed by a query:

`entity0 value0 entity1 value1 entity2 value2 QUERY queried_entity`

The target is the value bound to the queried entity. Train and holdout literal sequences remain unique and disjoint.

## Models

- `core_full_content`: recurrent control receives all tokens.
- `core_content_blind`: recurrent control receives no token embeddings.
- `mediation_final_only`: fixed sparse mediation topology, final loss only.
- `mediation_aux_annealed`: same topology with workspace auxiliary weight annealed from 0.5 to exactly zero over 200 steps.

The fixed mediation topology has three writer mechanisms and one query reader. Each binding pair is routed to one writer and one fixed workspace slot. Only the reader receives workspace broadcast. The spine reads neither embeddings nor workspace, and the output reads neither spine nor workspace directly; it reads only the final active reader state.

## Causal gates

Evaluation-only interventions include broadcast removal, workspace persistence removal, complete workspace-write removal, reader-inbox zeroing, recipient permutation, and output-path controls. A useful mediator must deteriorate under workspace-write and reader-inbox interventions after auxiliary supervision has fully withdrawn.

## Formal protocol

30 paired seeds, 400 updates, batch 8, 160 unique training sequences, 48 unique holdout sequences, 50,000 paired bootstrap replicates, deterministic CSV and SHA-256 manifests. No positive claim is allowed from a smoke run.
