# Post tuning development sweep

Status: **complete**. Expected trials: 15; missing: 0; rejected: 0.

Selection: mean validation accuracy ascending; mean validation NLL descending; stable profile_id ascending. ACC columns are percentages. NLL is mean negative log probability of the validation label (higher means a stronger attack).

| Dataset | Source budget | Actions | Profile | Valid/expected seeds | Validation ACC % ± sample SD | Validation NLL | Status |
|---|---:|---:|---|---:|---:|---:|---|
| citeseer | 0.1 | 60 | edit_control | 1/1 | 66.8246 ± undefined | 1.367947 | complete |
| citeseer | 0.1 | 60 | transfer_relu | 1/1 | 68.2464 ± undefined | 1.354206 | complete |
| citeseer | 0.1 | 60 | transfer_both | 1/1 | 68.7204 ± undefined | 1.326086 | complete |
| citeseer | 0.1 | 60 | edit_both | 1/1 | 69.1943 ± undefined | 1.297042 | complete |
| citeseer | 0.1 | 60 | transfer_residual | 1/1 | 70.6161 ± undefined | 1.258807 | complete |

## Paired ACC comparisons

Positive control-minus-Post ACC (percentage points) favors Post. ACC-only W/L/T does not count an NLL-only improvement as an ACC win. Intervals are unadjusted seed-level development intervals; n=1 is undefined.

| Dataset / actions | Post profile | Control profile / arm | Mode | Valid/observed/expected seeds | ACC difference mean ± SD (pp) | 95% CI (pp) | ACC-only W/L/T | Status |
|---|---|---|---|---:|---:|---|---|---|
| citeseer / 60 | edit_both | edit_control / endpoint_post | Post profile versus reference Post | 1/1/1 | -2.3697 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | edit_both | edit_both / endpoint_pre | matched configuration | 1/1/1 | 0.9479 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | edit_both | edit_both / parallel | matched configuration | 1/1/1 | -0.9479 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | edit_control | edit_control / endpoint_pre | matched configuration | 1/1/1 | 1.4218 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | edit_control | edit_control / parallel | matched configuration | 1/1/1 | 0.4739 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | transfer_both | edit_control / endpoint_post | Post profile versus reference Post | 1/1/1 | -1.8957 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | transfer_both | edit_both / endpoint_post | Post transfer policy versus original combined editor | 1/1/1 | 0.4739 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | transfer_both | transfer_both / endpoint_pre | matched configuration | 1/1/1 | 0.4739 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | transfer_both | transfer_both / parallel | matched configuration | 1/1/1 | 0.0000 ± undefined | undefined | 0/0/1 | complete |
| citeseer / 60 | transfer_relu | edit_control / endpoint_post | Post profile versus reference Post | 1/1/1 | -1.4218 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | transfer_relu | edit_both / endpoint_post | Post transfer policy versus original combined editor | 1/1/1 | 0.9479 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | transfer_relu | transfer_relu / endpoint_pre | matched configuration | 1/1/1 | 0.9479 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | transfer_relu | transfer_relu / parallel | matched configuration | 1/1/1 | 0.4739 ± undefined | undefined | 1/0/0 | complete |
| citeseer / 60 | transfer_residual | edit_control / endpoint_post | Post profile versus reference Post | 1/1/1 | -3.7915 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | transfer_residual | edit_both / endpoint_post | Post transfer policy versus original combined editor | 1/1/1 | -1.4218 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | transfer_residual | transfer_residual / endpoint_pre | matched configuration | 1/1/1 | -1.4218 ± undefined | undefined | 0/1/0 | complete |
| citeseer / 60 | transfer_residual | transfer_residual / parallel | matched configuration | 1/1/1 | -1.4218 ± undefined | undefined | 0/1/0 | complete |

paired_comparisons.csv and summary.json retain paired seed differences, ACC-only outcomes, sample SD and 95% intervals. Historical wins/losses/ties and outcome fields still use ACC first, then NLL for tied ACC; accuracy_wins/losses/ties and accuracy_outcome use ACC only. n counts observed pairs; valid_n is zero for rejected cells and their intervals are suppressed. Their remaining deltas and counts are audit data only. Partial or rejected comparisons cannot establish an advantage.

per_trial.csv records source and effective budgets, changed feature nodes/entries/L1/L2/Linf, successful writes, runtime, and coverage. Mean unique coverage averages cumulative distinct selected nodes over rounds; endpoint occupancy is the fraction of each round's four selections at its two action endpoints.

Fresh/repeated write fractions count first/repeat visits across the full trajectory. Coverage profiles additionally report fresh nodes in the unchanged diffusion pool and selected fresh nodes per round. A fresh3 quota shortfall means fewer than three fresh nodes in that pool before selection; it is not the final write-count deficit. When the original editgain profile is planned, new coverage Post profiles are also compared with its Post arm.

## Interpretation

- Development selection uses validation data only. Seeds 15–19 already have historical evaluations and are not a pristine holdout.
- A short prefix is not the full source-budget attack. One-seed results do not establish stable performance; sample SD is undefined at n=1.
- Post-only profiles versus baseline Pre/Parallel are exploratory. Matched configurations are required to attribute differences to graph timing.
- Paired 95% t intervals use seed-level ACC differences and are unadjusted, descriptive development intervals. They do not correct for searching many profiles or establish independent test-set superiority.
- Partial rankings retain valid observations, but any missing, rejected or unmatched requested trial suppresses every recommendation.

## Transfer policy factorial (ACC percentage points)

Anchor c=edit_both; r=transfer_relu; s=transfer_residual; b=transfer_both. Proxy=((c-r)+(s-b))/2; residual=((c-s)+(r-b))/2; interaction=r+s-c-b. Positive means lower attacked ACC. edit_control is the separate prior best reference, not a factorial cell. The proxy factor bundles ReLU architecture, fitting and independent initialization; it does not isolate activation alone.

| Dataset / actions | Arm | Effect | Valid/expected seeds | Mean ± SD (pp) | 95% CI | Status |
|---|---|---|---:|---|---|---|
| citeseer / 60 | endpoint_post | proxy | 1/1 | 1.421800947867304 ± None | None | complete |
| citeseer / 60 | endpoint_post | residual | 1/1 | -0.9478672985781955 ± None | None | complete |
| citeseer / 60 | endpoint_post | interaction | 1/1 | 0.9478672985781884 ± None | None | complete |
| citeseer / 60 | endpoint_pre | proxy | 1/1 | 0.4739336492891013 ± None | None | complete |
| citeseer / 60 | endpoint_pre | residual | 1/1 | 0.4739336492891013 ± None | None | complete |
| citeseer / 60 | endpoint_pre | interaction | 1/1 | -0.9478672985782026 ± None | None | complete |
| citeseer / 60 | parallel | proxy | 1/1 | 7.105427357601002e-15 ± None | None | complete |
| citeseer / 60 | parallel | residual | 1/1 | -0.4739336492891013 ± None | None | complete |
| citeseer / 60 | parallel | interaction | 1/1 | 0.9478672985782026 ± None | None | complete |

All policies use three real requests and four writes per action. Residual editing is a planned feature patch through the existing partial vocabulary, not full text replacement or verified semantic remapping. Unplanned coordinates remain unchanged. Gradients and candidate scores follow the arm graph. Actual selections and edit amplitudes may diverge with evolving features; per_trial.csv retains coverage, request counts, support changes and L1. The separate proxy is fitted on clean data with training labels only; it does not model subsequent victim retraining.
