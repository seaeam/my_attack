# Post tuning development sweep

Status: **complete**. Expected trials: 12; missing: 0; rejected: 0.

Selection: mean validation accuracy ascending; mean validation NLL descending; stable profile_id ascending. ACC columns are percentages. NLL is mean negative log probability of the validation label (higher means a stronger attack).

| Dataset | Source budget | Actions | Profile | Valid/expected seeds | Validation ACC % ± sample SD | Validation NLL | Status |
|---|---:|---:|---|---:|---:|---:|---|
| citeseer | 0.1 | 60 | edit_control | 1/1 | 66.8246 ± undefined | 1.367947 | complete |
| citeseer | 0.1 | 60 | edit_gain | 1/1 | 68.2464 ± undefined | 1.347628 | complete |
| citeseer | 0.1 | 60 | edit_words | 1/1 | 68.2464 ± undefined | 1.324456 | complete |
| citeseer | 0.1 | 60 | edit_both | 1/1 | 69.1943 ± undefined | 1.297042 | complete |

## Paired ACC comparisons

Positive control-minus-Post ACC (percentage points) favors Post. ACC-only W/L/T does not count an NLL-only improvement as an ACC win. Intervals are unadjusted seed-level development intervals; n=1 is undefined.

| Dataset / actions | Post profile | Control profile / arm | Mode | Valid/observed/expected seeds | ACC difference mean ± SD (pp) | 95% CI (pp) | ACC-only W/L/T | Status |
|---|---|---|---|---:|---:|---|---|---|
| citeseer / 60 | edit_both | edit_control / endpoint_post | Post profile versus reference Post | 1/1/1 | -2.3697 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | edit_both | edit_both / endpoint_pre | matched configuration | 1/1/1 | 0.9479 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | edit_both | edit_both / parallel | matched configuration | 1/1/1 | -0.9479 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | edit_control | edit_control / endpoint_pre | matched configuration | 1/1/1 | 1.4218 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | edit_control | edit_control / parallel | matched configuration | 1/1/1 | 0.4739 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | edit_gain | edit_control / endpoint_post | Post profile versus reference Post | 1/1/1 | -1.4218 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | edit_gain | edit_gain / endpoint_pre | matched configuration | 1/1/1 | 0.0000 ± undefined | undefined | 0/0/1 | complete |
| citeseer / 60 | edit_gain | edit_gain / parallel | matched configuration | 1/1/1 | 0.0000 ± undefined | undefined | 0/0/1 | complete |
| citeseer / 60 | edit_words | edit_control / endpoint_post | Post profile versus reference Post | 1/1/1 | -1.4218 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | edit_words | edit_words / endpoint_pre | matched configuration | 1/1/1 | 1.8957 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | edit_words | edit_words / parallel | matched configuration | 1/1/1 | -1.4218 ± undefined | undefined | 0/1/0 | complete |

paired_comparisons.csv and summary.json retain paired seed differences, ACC-only outcomes, sample SD and 95% intervals. Historical wins/losses/ties and outcome fields still use ACC first, then NLL for tied ACC; accuracy_wins/losses/ties and accuracy_outcome use ACC only. n counts observed pairs; valid_n is zero for rejected cells and their intervals are suppressed. Their remaining deltas and counts are audit data only. Partial or rejected comparisons cannot establish an advantage.

per_trial.csv records source and effective budgets, changed feature nodes/entries/L1/L2/Linf, successful writes, runtime, and coverage. Mean unique coverage averages cumulative distinct selected nodes over rounds; endpoint occupancy is the fraction of each round's four selections at its two action endpoints.

Fresh/repeated write fractions count first/repeat visits across the full trajectory. Coverage profiles additionally report fresh nodes in the unchanged diffusion pool and selected fresh nodes per round. A fresh3 quota shortfall means fewer than three fresh nodes in that pool before selection; it is not the final write-count deficit. When the original editgain profile is planned, new coverage Post profiles are also compared with its Post arm.

## Interpretation

- Development selection uses validation data only. Seeds 15–19 already have historical evaluations and are not a pristine holdout.
- A short prefix is not the full source-budget attack. One-seed results do not establish stable performance; sample SD is undefined at n=1.
- Post-only profiles versus baseline Pre/Parallel are exploratory. Matched configurations are required to attribute differences to graph timing.
- Paired 95% t intervals use seed-level ACC differences and are unadjusted, descriptive development intervals. They do not correct for searching many profiles or establish independent test-set superiority.
- Partial rankings retain valid observations, but any missing, rejected or unmatched requested trial suppresses every recommendation.

## Editor factorial (ACC percentage points)

c=edit_control, g=edit_gain, w=edit_words, b=edit_both. Gain=((c-g)+(w-b))/2; words=((c-w)+(g-b))/2; interaction=g+w-c-b. Positive favors the editing change. All contrasts are formed within seed and arm first. Incomplete matrices suppress effect estimates.

| Arm | Effect | Valid/expected seeds | Mean ± SD (pp) | 95% CI (pp) | Status |
|---|---|---:|---|---|---|
| endpoint_post | gain | 1/1 | -1.1848341232227497 ± None | None | complete |
| endpoint_post | words | 1/1 | -1.1848341232227497 ± None | None | complete |
| endpoint_post | interaction | 1/1 | 0.4739336492890942 ± None | None | complete |
| endpoint_pre | gain | 1/1 | 0.0 ± None | None | complete |
| endpoint_pre | words | 1/1 | -1.8957345971564052 ± None | None | complete |
| endpoint_pre | interaction | 1/1 | 0.0 ± None | None | complete |
| parallel | gain | 1/1 | -1.1848341232227497 ± None | None | complete |
| parallel | words | 1/1 | 0.2369668246445471 ± None | None | complete |
| parallel | interaction | 1/1 | -0.4739336492890942 ± None | None | complete |

New editor cells share three real requests/round (50 output tokens each), with edit_control as their same-budget reference. They do not reuse the historical one-request baseline. Editor gradients and candidate losses use the arm's diffusion graph; this evaluates the full local editor's graph state, not diffusion timing alone. Candidate duplicates, projection fractions and actual signed gains are retained in per_trial.csv and rounds.json.
