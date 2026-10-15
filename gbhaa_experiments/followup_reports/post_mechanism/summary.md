# Post tuning development sweep

Status: **complete**. Expected trials: 24; missing: 0; rejected: 0.

Selection: mean validation accuracy ascending; mean validation NLL descending; stable profile_id ascending. ACC columns are percentages. NLL is mean negative log probability of the validation label (higher means a stronger attack).

| Dataset | Source budget | Actions | Profile | Valid/expected seeds | Validation ACC % ± sample SD | Validation NLL | Status |
|---|---:|---:|---|---:|---:|---:|---|
| citeseer | 0.1 | 60 | baseline | 2/2 | 68.7204 ± 3.3512 | 1.355682 | complete |
| citeseer | 0.1 | 60 | frontier_editgain | 2/2 | 69.6682 ± 2.0107 | 1.366468 | complete |
| citeseer | 0.1 | 60 | balanced_frontier | 2/2 | 69.6682 ± 1.3405 | 1.324161 | complete |
| citeseer | 0.1 | 60 | frontier | 2/2 | 70.6161 ± 1.3405 | 1.339031 | complete |

## Paired ACC comparisons

Positive control-minus-Post ACC (percentage points) favors Post. ACC-only W/L/T does not count an NLL-only improvement as an ACC win. Intervals are unadjusted seed-level development intervals; n=1 is undefined.

| Dataset / actions | Post profile | Control profile / arm | Mode | Valid/observed/expected seeds | ACC difference mean ± SD (pp) | 95% CI (pp) | ACC-only W/L/T | Status |
|---|---|---|---|---:|---:|---|---|---|
| citeseer / 60 | balanced_frontier | baseline / endpoint_post | Post profile versus baseline Post | 2/2/2 | -0.9479 ± 2.0107 | [-19.0136, 17.1178] | 1/1/0 | complete |
| citeseer / 60 | balanced_frontier | balanced_frontier / endpoint_pre | matched configuration | 2/2/2 | 0.2370 ± 1.0054 | [-8.7959, 9.2698] | 1/1/0 | complete |
| citeseer / 60 | balanced_frontier | balanced_frontier / parallel | matched configuration | 2/2/2 | 0.0000 ± 0.0000 | [0.0000, 0.0000] | 0/0/2 | complete |
| citeseer / 60 | baseline | baseline / endpoint_pre | matched configuration | 2/2/2 | 1.1848 ± 3.0161 | [-25.9137, 28.2834] | 1/1/0 | complete |
| citeseer / 60 | baseline | baseline / parallel | matched configuration | 2/2/2 | -0.2370 ± 1.0054 | [-9.2698, 8.7959] | 1/1/0 | complete |
| citeseer / 60 | frontier | baseline / endpoint_post | Post profile versus baseline Post | 2/2/2 | -1.8957 ± 2.0107 | [-19.9614, 16.1700] | 0/2/0 | complete |
| citeseer / 60 | frontier | frontier / endpoint_pre | matched configuration | 2/2/2 | -0.9479 ± 0.6702 | [-6.9698, 5.0740] | 0/2/0 | complete |
| citeseer / 60 | frontier | frontier / parallel | matched configuration | 2/2/2 | -0.9479 ± 1.3405 | [-12.9917, 11.0959] | 0/1/1 | complete |
| citeseer / 60 | frontier_editgain | baseline / endpoint_post | Post profile versus baseline Post | 2/2/2 | -0.9479 ± 1.3405 | [-12.9917, 11.0959] | 0/1/1 | complete |
| citeseer / 60 | frontier_editgain | frontier_editgain / endpoint_pre | matched configuration | 2/2/2 | 1.1848 ± 1.0054 | [-7.8480, 10.2177] | 2/0/0 | complete |
| citeseer / 60 | frontier_editgain | frontier_editgain / parallel | matched configuration | 2/2/2 | -1.8957 ± 0.6702 | [-7.9176, 4.1262] | 0/2/0 | complete |

paired_comparisons.csv and summary.json retain paired seed differences, ACC-only outcomes, sample SD and 95% intervals. Historical wins/losses/ties and outcome fields still use ACC first, then NLL for tied ACC; accuracy_wins/losses/ties and accuracy_outcome use ACC only. n counts observed pairs; valid_n is zero for rejected cells and their intervals are suppressed. Their remaining deltas and counts are audit data only. Partial or rejected comparisons cannot establish an advantage.

per_trial.csv records source and effective budgets, changed feature nodes/entries/L1/L2/Linf, successful writes, runtime, and coverage. Mean unique coverage averages cumulative distinct selected nodes over rounds; endpoint occupancy is the fraction of each round's four selections at its two action endpoints.

## Interpretation

- Development selection uses validation data only. Seeds 15–19 already have historical evaluations and are not a pristine holdout.
- A short prefix is not the full source-budget attack. One-seed results do not establish stable performance; sample SD is undefined at n=1.
- Post-only profiles versus baseline Pre/Parallel are exploratory. Matched configurations are required to attribute differences to graph timing.
- Paired 95% t intervals use seed-level ACC differences and are unadjusted, descriptive development intervals. They do not correct for searching many profiles or establish independent test-set superiority.
- Partial rankings retain valid observations, but any missing, rejected or unmatched requested trial suppresses every recommendation.
