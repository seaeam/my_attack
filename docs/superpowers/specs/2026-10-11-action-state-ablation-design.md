# Matched pre/post structural-action ablation

## Request and scope

The user authorizes evaluating/correcting the experimental design and implementing
the code and launch commands now. They will execute the real experiments. Do not
edit the thesis, run a real dataset attack, contact an LLM, alter historical
results, or change the existing attack loop's attribute/edge ordering.

## Scientific question

Test whether diffusing on the graph after a particular edge flip improves
attribute localization relative to the same endpoints on the graph before that
flip. This is an empirical hypothesis, not a guaranteed proof of necessity.

Three arms share one saved, degree-source, attribute-weighted structural search
trajectory per dataset/budget/seed:

| Arm | Local restart | Diffusion graph |
|---|---|---|
| parallel | Two distinct nodes sampled uniformly by a separate per-round RNG | pre |
| endpoint_pre | The saved structural action's two endpoints | pre |
| endpoint_post | The same endpoints | post |

Parallel's sampling does not inspect or exclude the true endpoints. It can
coincide by chance; record that overlap. All arms use the same attribute-weighted
local operator, full node candidate universe, eligibility rules and stable node-ID
ties. Thus Parallel versus Pre isolates endpoint conditioning more cleanly than
changing to degree/random ranking at the same time.

## Matching and interpretation

Generate the structural trajectory once with frozen original structural
features and independent per-round structural RNG, without attribute generation.
Save original graph, action triples, final graph and input/config identities.
Replay those exact triples in each arm; no arm reruns endpoint search.

Each replay starts from the same original graph/features and surrogate. Reset the
internal training RNG each round, and train on original features and the shared
pre-flip graph. Only the local diffusion graph differs between Endpoint-Pre and
Endpoint-Post. Apply the flip before generation in every arm, and compute grouping
embeddings on the post-flip graph with the arm's current features in every arm.
This common grouping convention avoids additionally changing the grouping graph.
Pre/Parallel selection happens on the pre-flip snapshot; Post selection happens
after executing the flip. Every round executes precisely one flip.

Arms' current features may diverge after earlier choices. The final-ACC contrast
estimates the cumulative localization-policy effect under fixed structural
actions, not a one-step effect with identical intermediate attributes. Add
within-round diagnostics: on each arm's same current features/eligibility state,
compute the true-endpoint Pre and Post rankings and record score L1 difference,
selected-set overlap, zero-similarity edge, graph hashes and restart nodes.
These diagnostics do not themselves establish an accuracy improvement.

The experiment uses an explicit small matched attribute workload: 4 admitted
nodes per structural round, at most 10 visits per node, no distinct-node cap.
Reject runs that cannot fill a round or have generation/write failures. This
matches node-update opportunities (4 times the edge budget), not final distinct
nodes, changed feature entries or perturbation norms. Report those actual
quantities. Existing generation, grouping and projection rules are reused, with
round-local legacy template cache cleared and an additional persistent cache
keyed by the full LLM request/settings; equal requests across arms reuse replies.
Do not claim the backend is intrinsically deterministic just because seed and
temperature are set.

The base writer needs a completion callback exposing its real local counters.
Its default callback preserves the old failure rule. The new protocol overrides
only this callback: exactly four successful writes and zero failures are
required, while four successful no-op writes are valid and recorded. Otherwise
excluding a no-change round would select runs by the treatment's realized effect.
The old structural/attribute loop order remains byte-for-byte unchanged.

## Settings and runtime scope

- Defaults: Citeseer and Cora; seeds 15--19; structural budgets 0.05 and 0.10.
- Pubmed is opt-in, never auto-added. It may be memory expensive.
- 20 shared structural plans plus 60 attacked arms by default.
- Global source degree; alpha .15, 30 accumulated states; attribute weights on.
- Local endpoint/independent-pair AE-PR: restart .28, 60 updates, last-state readout.
- Reuse controlled follow-up solver/generator settings (level 2, miter 10,
  frozen structural features). Explicit 4-node per-round cap overrides ratio.
- Same split/target set and retrained GCN evaluation as the existing entrypoint;
  same clean/victim seed streams. This is a controlled GCN mechanism experiment,
  not a replacement for all main victim results.
- Use isolated output root `gbhaa_experiments/followup_results_action_state` and
  a new protocol ID. Plan-only default. `--execute` required for attacks.
- Allow single-dataset/single-seed pilot and selecting budget(s). Resume only
  exact source/input/environment/settings/trajectory identities with validated
  artifacts. Never overwrite complete results with changed configuration.
- Cache is scoped by experiment provenance/model/request; failed attempts are
  archived only with `--retry-failed`. Do not reuse legacy experiment outputs.

## Artifacts and summaries

Save commands before child launch, source/input/environment fingerprints,
trajectory hashes, selected nodes and restart nodes per round, graph-state
checks, generation request/cache counters, actual node visits/modified nodes,
changed entries, L1/L2/Linf change, original/final adjacency and features,
target-level predictions and recomputed ACC. Identity mismatches or incomplete
budgets exclude a run and its paired block. Save generation requests/replies in
the local request cache for auditability.

Summary defaults to complete 3-arm seed blocks only, grouped by dataset and
budget; never pool datasets or treat nodes/rounds as independent samples.
Report each arm's attacked ACC mean, sample SD and t-based 95% mean CI. Report
per-seed differences and mean/SD/t-based 95% CI for Pre minus Post (primary),
Parallel minus Post and Parallel minus Pre (secondary), in percentage points.
Positive means the second arm reduces ACC more. Label n<5 results preliminary;
do not manufacture zero-width intervals for n=1. The 5-seed t CI is approximate.
Optionally reuse existing exact paired sign-flip p-values; Holm adjust primary
contrasts across requested dataset/budget cells, with missing cells treated as
p=1. Secondary contrasts are descriptive.

## Acceptance

Offline tests must show true pre/post edge values for additions and deletions,
identical replayed trajectories, independent Parallel seed generation, same
eligibility/budget rules, different rankings where an edge affects diffusion,
strict artifact identities, paired statistics, safe resume and no-run dry-run.
Tests use synthetic graphs and fake LLM replies only. Existing factorial and PR
protocol regression tests must keep passing. Thesis checksum stays unchanged.
