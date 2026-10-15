# ReLU editor objective pilot

## Scope and evidence

User authorizes a quick matched Post/Pre/Parallel comparison, based on `transfer_relu`, with code and launcher only. No thesis changes, real experiment execution, test-label selection, forced winner, changed comparator budgets, or changed structural search. Existing 40 profile identities and historical artifacts remain valid.

Read-only same-state replay of seed 15 `transfer_relu__endpoint_post` found 62.74% of positive pseudo-NLL increments on targets already predicted away from their fixed source pseudo-labels. A capped margin changes 92/240 candidate choices (slot counts 85/10/145 versus 3/1/236). This is evidence for a hypothesis, not a victim-accuracy estimate or a new experiment result.

## Four cells

Baseline `transfer_relu` remains byte-identical. New cells `objective_margin`, `objective_local`, `objective_both` inherit its ReLU proxy, full replacement, signed-gradient word plan, actual projected candidate selection, three requests/candidates and four writes per action.

1. Baseline: uniform mean pseudo-NLL.
2. Margin: maximize `-relu(logp[pseudo] - max_other(logp))`, uniform over the unchanged nontraining target set. The score is upper-bounded at zero, with zero derivative once the pseudo-class loses. It does not use true nontraining labels.
3. Local: unchanged pseudo-NLL with weights `0.5/N + 0.5 * diffusion[target]/sum(diffusion[target])`. If target diffusion mass is at most 1e-12, use uniform weights and record the fallback.
4. Both: local weights and capped margin.

Weights use the current round-start features, this arm's graph and restarts, and existing AE-PR settings. They are frozen for word gradients and every sequential candidate score during the round. The scalar objective used for gradients and actual post-projection candidate comparison is identical. Score ties retain the existing stable candidate rule. No no-op candidate, eligibility restriction, additional query or candidate is introduced.

## Matching and provenance

Post uses the executed post-flip graph and true endpoints; Pre uses the pre-flip graph and identical endpoints; Parallel uses the pre-flip graph and existing independently seeded restarts. Parallel must not inherit actual-endpoint weights. Each arm follows its own evolving features. Pseudo-labels and target IDs remain fixed from the source surrogate. Every profile shares the same clean ReLU checkpoint, seeds, splits, structure prefix, projection, write/query budget, and victim fitting protocol. Objectives intentionally use different arm-local states, the treatment being compared.

Record objective version/type, local mix, restart IDs, target weight/hash, fallback/mass/effective count. Audit independently derives restarts and graph from the action/arm, recomputes weights, gradients, candidate bank and chosen writes. Existing traces must be unchanged. Exact profile/operator schemas reject unknown objective fields and invalid mixes/policies.

## Delivery and analysis

New preset `objective`, anchored on `transfer_relu`, with an isolated `post_objective_results` root. Default launcher: Citeseer, paired seeds 15 and 16, 60 actions, four configurations times three arms = 24 short trials. `--seeds 15` yields 12 trials for smoke screening; 15–19 is the follow-up. Preview creates no output, execution requires `--execute`, retry uses `--retry-failed`.

Report normal validation ACC/NLL ranking, paired ACC differences with sample SD and t-based 95% CI, factorial effects (margin/local/interaction), and joint Post advantage against both comparators per seed. Positive Pre-minus-Post and Parallel-minus-Post means stronger Post. Ties are not strict wins; report the worst comparator and worst seed. Missing/rejected planned cells suppress aggregate claims and recommendations. A two-seed development result cannot establish general stability, especially after repeated selection on these data.

## Verification

Analytic objective/gradient and weight tests; wrong graph/restart/weight tampering; real synthetic replay through all arms; public 24-cell artifact loader and summary; missing results and ties; no-I/O launcher preview. Run relevant existing offline tests. Compare all 40 old profiles/operators/runtime settings, audit representative historical editor results without fitting/generation, and verify protected file hashes.
