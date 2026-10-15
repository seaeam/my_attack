# Post tuning development sweep

Status: **complete**. Expected trials: 12; missing: 0; rejected: 0.

Selection: mean validation accuracy ascending; mean validation NLL descending; stable profile_id ascending. ACC columns are percentages. NLL is mean negative log probability of the validation label (higher means a stronger attack).

| Dataset | Source budget | Actions | Profile | Valid/expected seeds | Validation ACC % ± sample SD | Validation NLL | Status |
|---|---:|---:|---|---:|---:|---:|---|
| citeseer | 0.1 | 60 | alpha050 | 1/1 | 66.3507 ± undefined | 1.345207 | complete |
| citeseer | 0.1 | 60 | baseline | 1/1 | 66.3507 ± undefined | 1.333904 | complete |
| citeseer | 0.1 | 60 | diffused | 1/1 | 66.8246 ± undefined | 1.326350 | complete |
| citeseer | 0.1 | 60 | floor025 | 1/1 | 67.2986 ± undefined | 1.359099 | complete |
| citeseer | 0.1 | 60 | coverage05 | 1/1 | 67.7725 ± undefined | 1.317068 | complete |
| citeseer | 0.1 | 60 | coverage10 | 1/1 | 67.7725 ± undefined | 1.274151 | complete |
| citeseer | 0.1 | 60 | alpha015 | 1/1 | 68.2464 ± undefined | 1.315720 | complete |
| citeseer | 0.1 | 60 | floor010 | 1/1 | 68.2464 ± undefined | 1.296570 | complete |
| citeseer | 0.1 | 60 | combined | 1/1 | 68.2464 ± undefined | 1.292863 | complete |
| citeseer | 0.1 | 60 | alpha010 | 1/1 | 68.7204 ± undefined | 1.334707 | complete |

Comparisons in summary.json retain every paired seed difference, win/loss/tie, and sample SD. Positive control-minus-Post ACC (percentage points) and positive Post-minus-control NLL favor Post. Wins use ACC first, then NLL for tied ACC.

per_trial.csv records source and effective budgets, changed feature nodes/entries/L1/L2/Linf, successful writes, runtime, and coverage. Mean unique coverage averages cumulative distinct selected nodes over rounds; endpoint occupancy is the fraction of each round's four selections at its two action endpoints.

## Interpretation

- Development selection uses validation data only. Seeds 15–19 already have historical evaluations and are not a pristine holdout.
- A short prefix is not the full source-budget attack. One-seed results do not establish stable performance; sample SD is undefined at n=1.
- Post-only profiles versus baseline Pre/Parallel are exploratory. Matched configurations are required to attribute differences to graph timing.
- Partial rankings retain valid observations, but any missing, rejected or unmatched requested trial suppresses every recommendation.
