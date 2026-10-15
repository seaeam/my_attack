# Post tuning development sweep

Status: **complete**. Expected trials: 16; missing: 0; rejected: 0.

Selection: mean validation accuracy ascending; mean validation NLL descending; stable profile_id ascending. ACC columns are percentages. NLL is mean negative log probability of the validation label (higher means a stronger attack).

| Dataset | Source budget | Actions | Profile | Valid/expected seeds | Validation ACC % ± sample SD | Validation NLL | Status |
|---|---:|---:|---|---:|---:|---:|---|
| citeseer | 0.1 | 120 | baseline | 4/4 | 69.6682 ± 2.3538 | 1.310111 | complete |
| citeseer | 0.1 | 120 | alpha050 | 4/4 | 70.6161 ± 2.8436 | 1.306340 | complete |

## Paired ACC comparisons

Positive control-minus-Post ACC (percentage points) favors Post. ACC-only W/L/T does not count an NLL-only improvement as an ACC win. Intervals are unadjusted seed-level development intervals; n=1 is undefined.

| Dataset / actions | Post profile | Control profile / arm | Mode | Valid/observed/expected seeds | ACC difference mean ± SD (pp) | 95% CI (pp) | ACC-only W/L/T | Status |
|---|---|---|---|---:|---:|---|---|---|
| citeseer / 120 | alpha050 | baseline / endpoint_post | Post profile versus baseline Post | 4/4/4 | -0.9479 ± 0.6702 | [-2.0144, 0.1186] | 0/4/0 | complete |
| citeseer / 120 | alpha050 | alpha050 / endpoint_pre | matched configuration | 4/4/4 | -0.1185 ± 1.6130 | [-2.6851, 2.4482] | 2/1/1 | complete |
| citeseer / 120 | alpha050 | alpha050 / parallel | matched configuration | 4/4/4 | -0.5924 ± 1.5173 | [-3.0068, 1.8220] | 1/1/2 | complete |

paired_comparisons.csv and summary.json retain paired seed differences, ACC-only outcomes, sample SD and 95% intervals. Historical wins/losses/ties and outcome fields still use ACC first, then NLL for tied ACC; accuracy_wins/losses/ties and accuracy_outcome use ACC only. n counts observed pairs; valid_n is zero for rejected cells and their intervals are suppressed. Their remaining deltas and counts are audit data only. Partial or rejected comparisons cannot establish an advantage.

per_trial.csv records source and effective budgets, changed feature nodes/entries/L1/L2/Linf, successful writes, runtime, and coverage. Mean unique coverage averages cumulative distinct selected nodes over rounds; endpoint occupancy is the fraction of each round's four selections at its two action endpoints.

## Interpretation

- Development selection uses validation data only. Seeds 15–19 already have historical evaluations and are not a pristine holdout.
- A short prefix is not the full source-budget attack. One-seed results do not establish stable performance; sample SD is undefined at n=1.
- Post-only profiles versus baseline Pre/Parallel are exploratory. Matched configurations are required to attribute differences to graph timing.
- Paired 95% t intervals use seed-level ACC differences and are unadjusted, descriptive development intervals. They do not correct for searching many profiles or establish independent test-set superiority.
- Partial rankings retain valid observations, but any missing, rejected or unmatched requested trial suppresses every recommendation.
