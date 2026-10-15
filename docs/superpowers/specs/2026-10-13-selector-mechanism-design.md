# Endpoint selector mechanism experiment

## Scope and evidence

User explicitly requests deeper analysis, selection-mechanism changes, a quick comparison script to run themselves, and no thesis changes. No real training/LLM experiments will be launched by the assistant. Existing results and default algorithms stay intact.

Saved Citeseer seeds 16–19, 120-action runs show: baseline Post and alpha050 endpoint arms always select both endpoints (half of four writes); alpha050 Post same-state pre/post probes change the selected set in 43/480 rounds. All 480 rounds use one four-node text group, bypassing embedding-based subdivision. Selected-node mean pairwise cosine increases from approximately .19 before writes to .83 after. Parallel covers more distinct nodes. These are descriptive observations, not proof of why ACC changes.

## Alternatives and selected design

We considered graph-difference ranking (requires both graphs and changes the meaning of Pre), changing generation/grouping (outside the requested selector isolation), and single-graph selector changes (selected). The new algorithms only receive their own arm graph and restart pair. No Post-derived candidate pool is shared with controls; no validation/test labels or outcomes enter selection.

Four profiles form the new `mechanism` preset. Baseline remains unchanged at alpha=.28, N=60, floor=0, gamma=0, last readout. Three explicit method variants share these settings and all generation/projection/node-write budgets:

1. `frontier`: prioritize legal non-restart nodes with positive diffusion mass; use original stable diffusion ordering for the remainder. This removes obligatory endpoint expenditure when enough reachable alternatives exist. If fewer than four non-restart nodes have positive mass, documented fallback may include restarts; never select zero-mass arbitrary nodes ahead of reachable restarts.
2. `balanced_frontier`: solve two single-source diffusions separately and interleave their positive legal non-restart rankings, skipping duplicates, to distribute the four selections across both sides. Sort the two restart IDs first for orientation invariance. Exhausted sides fall back to the other side and then the original full ranking. Return a permutation of the entire node universe.
3. `frontier_editgain`: take up to 32 positive legal non-restart candidates from the same graph's diffusion. Rerank by an approximate feature-edit gain from a fixed clean surrogate: maximize NLL of clean surrogate argmax pseudo-labels over all non-training nodes. Gradients use current features and only this arm's graph. Gain sums beneficial removals `max(-g*x,0)` over feature dimensions plus the largest up-to-70 positive gradients for currently absent vocabulary-supported columns, assuming unit additions. This is a first-order opportunity proxy, not realized LLM edits or guaranteed damage. Ties use diffusion score then node ID. If all candidate gains are zero, use frontier ordering. Actual generation and projection are unchanged.

The clean surrogate is reused; no extra fitting, victim evaluation, or LLM candidate search occurs. Save its pseudo-labels, gradient target IDs and vocabulary width for audit. Restore model training mode after gradient computation and do not modify model parameters or `.grad` fields.

## Integration and audit

Keep legacy profile dictionaries exactly unchanged. Only new profiles add `selector_rule` and `selector_pool_size`; strict validation permits these together only with known rules and a bounded integer pool >=4. Operator metadata explicitly labels `endpoint-frontier-v1` and the extra surrogate assumption. Existing protocol remains the same replay/evaluation protocol, with distinct catalog identities and new output root.

Add a small mechanism module for ordering, edit potential and gradient calculation. TuningAttack dispatches only new profiles to it. Pre/post diagnostic probes use the exact same mechanism with their own graph; Parallel uses independent restarts on pre. Existing arm graph hashes, selected/probe equality, visit budget and feature-state audit remain valid.

Save mechanism traces for pre/post and actual Parallel selection: graph/restart identity, candidate pool, positive-mass size, selected nodes, gains where applicable, and fallback information. Validator checks these against selected nodes, arm identity and deterministic selection from stored scores/gains, plus pseudo-label/target artifact identities. This validates dispatch/ranking evidence, not a claim that stored gradients were independently retrained.

## Quick run and reporting

Default quick run: Citeseer, seeds 15 and 16, 60-action prefixes, 4 profiles × 3 arms × 2 seeds = 24 short trials. Optional `--seeds 15` produces 12 trials. New `post_mechanism_results` and report directory isolate source fingerprints and old results. Default launcher previews; --execute required. Reuse validation-only ranking and complete-matrix reporting; two development seeds cannot establish stable superiority.

## Tests

Verify legacy catalog compatibility; removal of seed domination with positive support; isolate/fewer-candidate fallback; deterministic two-sided allocation and restart-swap invariance; visits applied before pools; edit opportunity arithmetic; directional finite-difference gradient check on a tiny model; no parameter/gradient/mode mutation; no ground-truth label inputs; actual three-arm graph dispatch; artifact tamper rejection; dry-run job counts and summary paths. Run all affected regression suites without network or training.
