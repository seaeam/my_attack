# Edit-gain coverage experiment

## Authorized scope and evidence

User requests optimizing frontier_editgain's repetition/coverage, implementing a quick matched comparison script for the user to run, and no thesis changes. No real training, attack or LLM calls by the assistant. Keep prior results, profile identities, old main-loop order, generation and four-write/ten-visit budgets unchanged.

Saved Post traces (Citeseer seeds 15/16, 60 rounds) have on average 26.52/26.55 fresh candidates in their 32-node diffusion pool. In 53/120 rounds (25/60 and 28/60), at least two fresh candidates exist but fewer than two are selected. Median best repeated-node gain divided by best fresh-node gain is 2.11/2.08, versus .95/.95 for Parallel. Thus investigate ranking preference before expanding the local pool. ACC benefit remains a hypothesis.

## Alternatives and chosen design

Considered pool expansion, hard visit caps/cooldowns, and within-pool coverage ranking. Select the third: the pool already contains fresh alternatives; stronger global eligibility constraints could make the workload infeasible and confound the visit budget. Two new catalog profiles at fixed alpha=.28, N=60, floor=0 and pool size=32:

- `frontier_editgain_decay`: score pool nodes by gain/(1+prior visits); tie-break by diffusion descending then ID ascending. Full stable frontier/diffusion fallback is unchanged. No mandatory coverage promise.
- `frontier_editgain_fresh3`: from the same original pool, choose up to three never-visited nodes by raw gain, diffusion, ID; then fill remaining slots by that pool's raw-gain order. If fewer than three fresh candidates exist, reserve those available and record shortfall; fill all four writes via existing fallback. No pool expansion, no additional generation calls, no global threshold change. Even when all gains are zero, retain the fresh quota and rank within each class by diffusion/ID; explicitly document this distinction from legacy zero-gain fallback.

Use only each arm's own graph, restart pair and prior visit history. Immutable clean surrogate/pseudo-labels/gradient targets and gradients are unchanged, with all guided rules included in cross-arm audits. Pre must never consume Post's pool or visit history. Same rules run on Post, Pre and Parallel.

Return a full node permutation with four first legal nodes. Reserve fresh choices only in the first four; append remaining pool nodes, remaining frontier, legal diffusion fallback and ineligible tail without duplicates. All legacy ordering/info dictionaries remain byte-for-byte equivalent for legacy inputs.

## Provenance, auditing, and reporting

Preserve all 31 existing profile dictionaries and operator metadata. Add new profiles in a separate catalog group, leaving the old mechanism preset at 24 matched two-seed trials. The new coverage preset has baseline, original frontier_editgain and the two new profiles: 24 default trials (seeds 15/16, 60 actions), 12 with one seed, separate post_coverage_results root.

New rule traces add fresh_pool_count, selected_fresh_count, selected_prior_visits, selected_adjusted_gains; fresh3 also requested/effective quota and shortfall. Common saved states/gains graph audits remain, with recomputed diffusion as authoritative ordering. Replay visits are reconstructed from actual writes, so tampered history-derived diagnostics fail validation. New traces record selector version endpoint-frontier-coverage-v1; legacy version remains unchanged.

Summary reports new and repeated write fractions, final coverage, fresh quota shortfalls, and compares new Post with original frontier_editgain Post as well as baseline Post and same-profile Pre/Parallel. Numerical rankings still use validation ACC, never force Post to win. Small repeated development seeds do not establish stable or test-set superiority.

## Verification

Tests first: divisor uses prior visits; quota reserves fresh nodes despite larger repeated gains; bounded pool unchanged; all-zero, scarce/no fresh nodes, fewer-than-four positive nodes, max-visit eligibility and ties; full permutation and budget; legacy profiles unchanged. Real synthetic replay and full artifact loader cover new rules on all three arms; tampering with fresh/visit diagnostics rejected. Actual launcher previews do not create result directories. Freeze old catalog snapshot before edits and revalidate legacy/new-mechanism saved artifact compatibility. No thesis edits or real experiment execution.
