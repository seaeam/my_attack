# Post tuning development sweep

Status: **complete**. Expected trials: 30; missing: 0; rejected: 0.

Selection: mean validation accuracy ascending; mean validation NLL descending; stable profile_id ascending. ACC columns are percentages. NLL is mean negative log probability of the validation label (higher means a stronger attack).

| Dataset | Source budget | Actions | Profile | Valid/expected seeds | Validation ACC % ± sample SD | Validation NLL | Status |
|---|---:|---:|---|---:|---:|---:|---|
| citeseer | 0.1 | 60 | alpha050 | 1/1 | 66.3507 ± undefined | 1.345207 | complete |
| citeseer | 0.1 | 60 | baseline | 1/1 | 66.3507 ± undefined | 1.333904 | complete |
| citeseer | 0.1 | 60 | a50_f01 | 1/1 | 67.2986 ± undefined | 1.346419 | complete |
| citeseer | 0.1 | 60 | a50_f03_diffused | 1/1 | 67.2986 ± undefined | 1.342331 | complete |
| citeseer | 0.1 | 60 | a50_f05 | 1/1 | 67.2986 ± undefined | 1.336544 | complete |
| citeseer | 0.1 | 60 | alpha060 | 1/1 | 67.2986 ± undefined | 1.312530 | complete |
| citeseer | 0.1 | 60 | alpha040 | 1/1 | 67.2986 ± undefined | 1.304395 | complete |
| citeseer | 0.1 | 60 | a50_coverage025 | 1/1 | 68.7204 ± undefined | 1.321271 | complete |
| citeseer | 0.1 | 60 | a50_diffused | 1/1 | 68.7204 ± undefined | 1.318750 | complete |
| citeseer | 0.1 | 60 | a50_f03 | 1/1 | 68.7204 ± undefined | 1.312576 | complete |

## Paired ACC comparisons

Positive control-minus-Post ACC (percentage points) favors Post. ACC-only W/L/T does not count an NLL-only improvement as an ACC win. Intervals are unadjusted seed-level development intervals; n=1 is undefined.

| Dataset / actions | Post profile | Control profile / arm | Mode | Valid/observed/expected seeds | ACC difference mean ± SD (pp) | 95% CI (pp) | ACC-only W/L/T | Status |
|---|---|---|---|---:|---:|---|---|---|
| citeseer / 60 | a50_coverage025 | baseline / endpoint_post | Post profile versus baseline Post | 1/1/1 | -2.3697 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_coverage025 | a50_coverage025 / endpoint_pre | matched configuration | 1/1/1 | -0.4739 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_coverage025 | a50_coverage025 / parallel | matched configuration | 1/1/1 | -0.4739 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_diffused | baseline / endpoint_post | Post profile versus baseline Post | 1/1/1 | -2.3697 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_diffused | a50_diffused / endpoint_pre | matched configuration | 1/1/1 | -0.9479 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_diffused | a50_diffused / parallel | matched configuration | 1/1/1 | 0.4739 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | a50_f01 | baseline / endpoint_post | Post profile versus baseline Post | 1/1/1 | -0.9479 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_f01 | a50_f01 / endpoint_pre | matched configuration | 1/1/1 | 0.9479 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | a50_f01 | a50_f01 / parallel | matched configuration | 1/1/1 | 0.9479 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | a50_f03 | baseline / endpoint_post | Post profile versus baseline Post | 1/1/1 | -2.3697 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_f03 | a50_f03 / endpoint_pre | matched configuration | 1/1/1 | -0.9479 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_f03 | a50_f03 / parallel | matched configuration | 1/1/1 | 0.0000 ± undefined | undefined | 0/0/1 | complete |
| citeseer / 60 | a50_f03_diffused | baseline / endpoint_post | Post profile versus baseline Post | 1/1/1 | -0.9479 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_f03_diffused | a50_f03_diffused / endpoint_pre | matched configuration | 1/1/1 | 0.0000 ± undefined | undefined | 0/0/1 | complete |
| citeseer / 60 | a50_f03_diffused | a50_f03_diffused / parallel | matched configuration | 1/1/1 | 0.9479 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | a50_f05 | baseline / endpoint_post | Post profile versus baseline Post | 1/1/1 | -0.9479 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | a50_f05 | a50_f05 / endpoint_pre | matched configuration | 1/1/1 | 1.4218 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | a50_f05 | a50_f05 / parallel | matched configuration | 1/1/1 | 0.9479 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | alpha040 | baseline / endpoint_post | Post profile versus baseline Post | 1/1/1 | -0.9479 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | alpha040 | alpha040 / endpoint_pre | matched configuration | 1/1/1 | 0.0000 ± undefined | undefined | 0/0/1 | complete |
| citeseer / 60 | alpha040 | alpha040 / parallel | matched configuration | 1/1/1 | 0.0000 ± undefined | undefined | 0/0/1 | complete |
| citeseer / 60 | alpha050 | baseline / endpoint_post | Post profile versus baseline Post | 1/1/1 | 0.0000 ± undefined | undefined | 0/0/1 | complete |
| citeseer / 60 | alpha050 | alpha050 / endpoint_pre | matched configuration | 1/1/1 | 2.3697 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | alpha050 | alpha050 / parallel | matched configuration | 1/1/1 | 0.9479 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | alpha060 | baseline / endpoint_post | Post profile versus baseline Post | 1/1/1 | -0.9479 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | alpha060 | alpha060 / endpoint_pre | matched configuration | 1/1/1 | 1.8957 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | alpha060 | alpha060 / parallel | matched configuration | 1/1/1 | 0.4739 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | baseline | baseline / endpoint_pre | matched configuration | 1/1/1 | 3.3175 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | baseline | baseline / parallel | matched configuration | 1/1/1 | 0.4739 ± undefined | undefined | 1/0/0 | complete |

paired_comparisons.csv and summary.json retain paired seed differences, ACC-only outcomes, sample SD and 95% intervals. Historical wins/losses/ties and outcome fields still use ACC first, then NLL for tied ACC; accuracy_wins/losses/ties and accuracy_outcome use ACC only. n counts observed pairs; valid_n is zero for rejected cells and their intervals are suppressed. Their remaining deltas and counts are audit data only. Partial or rejected comparisons cannot establish an advantage.

per_trial.csv records source and effective budgets, changed feature nodes/entries/L1/L2/Linf, successful writes, runtime, and coverage. Mean unique coverage averages cumulative distinct selected nodes over rounds; endpoint occupancy is the fraction of each round's four selections at its two action endpoints.

## Interpretation

- Development selection uses validation data only. Seeds 15–19 already have historical evaluations and are not a pristine holdout.
- A short prefix is not the full source-budget attack. One-seed results do not establish stable performance; sample SD is undefined at n=1.
- Post-only profiles versus baseline Pre/Parallel are exploratory. Matched configurations are required to attribute differences to graph timing.
- Paired 95% t intervals use seed-level ACC differences and are unadjusted, descriptive development intervals. They do not correct for searching many profiles or establish independent test-set superiority.
- Partial rankings retain valid observations, but any missing, rejected or unmatched requested trial suppresses every recommendation.
