# Endpoint-Post quick optimization design

## Objective and scope

Implement a user-run, small development sweep that seeks stronger Endpoint-Post
attack performance. Do not promise a winner, alter controls to make them weaker,
replace historical results, edit the manuscript, or run real attacks/LLM calls.
User explicitly authorizes implementation; no additional approval or commit is
needed. Preserve existing files and old loop behavior.

Default: Citeseer, seed 15, first 60 actions from the existing 10% shared
structure trajectory. Ten Post profiles plus baseline Pre and Parallel: twelve
short trials. Dry run by default. Record source budget and actual prefix budget
separately. Keep four node updates/round and max ten visits/node throughout.

## Search

Quick profiles: baseline; restart .10/.15/.50; edge-weight floor .10/.25;
visit penalties .5/1; propagated readout; and a combined profile. Floor applies
to every existing edge equally, w=floor+(1-floor)*clipped_cosine. Coverage divides
ranking scores by (1+visits)^gamma. Propagated readout removes direct restart
mass max(state-alpha*q,0). These additions are explicitly method variants.

An opt-in extended catalog also covers additional restart/floor combinations,
longer generation, two-node text groups, and
projection parameters. Generation/projection changes are explicitly labeled
and are not claims about the isolated post-flip mechanism. No meaningless grid
over ignored knobs such as topk_ratio with a fixed four-node cap. Do not include
scores that inspect both pre/post graphs: they would leak post information into
the Pre control. All catalog profiles are compatible with matched-controls.
Extended generation max_tokens is applied after inherited initialization and
before generation; the live value is included in cache keys and metadata.

## Architecture

New protocol `gbca-post-tuning-v1`, isolated result/report roots. New modules:
post_tuning_protocol (profile catalog, jobs, configurable scorer),
entrypoints/meta_post_tuning (subclass replay + validation-only evaluation),
post_tuning_validation (artifact checks), run_post_tuning (dry-run, execution,
resume), summarize_post_tuning (development rankings and recommendations).

Bootstrap meta.py with a run_name other than __main__: reuse data loading,
surrogate fitting and construction but do not call the legacy evaluation main.
Suppress the bootstrap's fixed surrogate test diagnostic from tuning output.
Use namespace idx_train/idx_val/idx_test; require disjoint nonempty sets. Train
victim on idx_train only; rank using idx_val ACC, then validation NLL (higher
loss wins ties), never held-out ACC. Save only validation predictions. Historical
seeds 15–19 were already evaluated, so this is development, not pristine holdout.

Subclass ActionStateAttack: keep inherited replay, generation/projection,
strict four-write completion and all graph/feature artifacts. Override options,
metadata, plan loading and scorer. Original controls use baseline configuration;
an optional matched-controls mode applies every selected configuration to all
three arms, for a small follow-up confirmation. Allow seed 16 with saved plans;
do not claim it is a new untouched test. Missing saved plans fail with an exact
path; no automatic expensive planning or fallback to unrelated actions.

## Identity and integrity

Read parent structure artifacts under their original protocol; verify saved
fingerprint, graph/action replay and provenance. Verify input graph/features,
labels/splits against current bootstrap. Slice the same ordered prefix, record
parent fingerprint/full action hash and prefix action hash. All profiles start
the original feature state. Current code/input/environment provenance is separate
from parent provenance. Preserve old artifacts as read-only.

Reuse the complete-request cache in a new tuning provenance namespace across
profiles. Generation max_tokens is part of the actual request key. Validate
each referenced cache response and request setting. Do not silently omit failed
trials or admit a recommendation from incomplete/unmatched blocks. Partial
leaderboards may display status; a recommendation requires all requested trials
on all requested seeds to pass. Sort by mean validation ACC ascending, mean
validation NLL descending, stable profile ID. All outcomes retained.

## Reporting and budget

Report per-trial validation ACC/NLL, actual changed nodes/entries/L1/L2/Linf,
successful writes, runtime, mean unique coverage and endpoint occupancy. Include
baseline Post improvement and available Pre/Parallel differences. Explicitly
label Post-only tuning vs baseline controls exploratory; matched-controls mode
is required for attributing improvements to the graph timing. Keep original
60-run results unchanged. One-seed short-prefix rankings are not stable formal
claims and cannot be relabeled as full 10% results.

## Tests

Synthetic only: baseline scorer parity, parameters alter scores/selection,
zero-weight edge becomes active with floor, coverage/readout logic,
prefix hashes/replay, real writer callback, validation-only evaluator, split
disjointness and test-label invariance, strict plan compatibility, true config
recording, cache settings, no-op-success vs failure, exact resume, incomplete
recommendation rejection, default twelve commands, optional matched controls,
and end-to-end tiny synthetic artifact validation/summary.
