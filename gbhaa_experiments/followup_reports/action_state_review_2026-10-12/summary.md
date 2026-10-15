# Matched action-state experiment

Complete three-arm seed blocks: 20; missing stages: 0; rejected records/blocks: 0.

The sampling unit is a complete model/attack seed. Datasets and budgets are summarized separately.
ACC is recomputed from target-level predictions. Means, sample SDs and t-based 95% mean CIs are reported in percent; paired differences are percentage points.
n < 5 is preliminary; SD and CI are undefined at n = 1. The five-seed t interval is approximate.
Pre minus Post is primary; Parallel minus Post and Parallel minus Pre are descriptive secondary contrasts. Positive means the second arm has lower attacked ACC.
The primary paired sign-flip test assumes sign exchangeability under the null. Holm covers all requested dataset/budget cells; missing cells use p = 1 only for correction and are not measurements.
The comparison estimates cumulative localization-policy effects under a fixed structural plan. Intermediate features can diverge between arms; this does not prove that post-action diffusion is necessary.
Four successful node-update opportunities per structural action are matched. Actual changed nodes, entries, visits and L1/L2/Linf norms appear in per_seed.csv; successful no-op writes remain eligible.

| Dataset | Budget | Arm | n | ACC (%) mean ± SD | 95% mean CI | Status |
|---|---:|---|---:|---:|---|---|
| citeseer | 0.05 | parallel | 5 | 67.3723 ± 1.5283 | [65.4747, 69.2699] | five or more seeds |
| citeseer | 0.05 | endpoint_pre | 5 | 68.4992 ± 1.8709 | [66.1762, 70.8222] | five or more seeds |
| citeseer | 0.05 | endpoint_post | 5 | 68.7098 ± 2.3111 | [65.8402, 71.5795] | five or more seeds |
| citeseer | 0.1 | parallel | 5 | 62.4539 ± 1.5697 | [60.5049, 64.4029] | five or more seeds |
| citeseer | 0.1 | endpoint_pre | 5 | 64.5919 ± 0.8895 | [63.4875, 65.6963] | five or more seeds |
| citeseer | 0.1 | endpoint_post | 5 | 64.2865 ± 0.7635 | [63.3385, 65.2344] | five or more seeds |
| cora | 0.05 | parallel | 5 | 78.0241 ± 0.9327 | [76.8661, 79.1822] | five or more seeds |
| cora | 0.05 | endpoint_pre | 5 | 77.6486 ± 1.1054 | [76.2761, 79.0212] | five or more seeds |
| cora | 0.05 | endpoint_post | 5 | 77.2910 ± 1.1054 | [75.9184, 78.6636] | five or more seeds |
| cora | 0.1 | parallel | 5 | 71.8462 ± 1.2622 | [70.2790, 73.4134] | five or more seeds |
| cora | 0.1 | endpoint_pre | 5 | 72.3827 ± 1.4438 | [70.5899, 74.1754] | five or more seeds |
| cora | 0.1 | endpoint_post | 5 | 72.5793 ± 1.5240 | [70.6870, 74.4717] | five or more seeds |

| Dataset | Budget | Paired contrast | Role | n | Difference (pp) | 95% mean CI | Holm p |
|---|---:|---|---|---:|---:|---|---:|
| citeseer | 0.05 | pre_minus_post | primary | 5 | -0.2106 | [-1.5249, 1.1036] | 0.75 |
| citeseer | 0.05 | parallel_minus_post | secondary | 5 | -1.3375 | [-3.2814, 0.6063] | descriptive |
| citeseer | 0.05 | parallel_minus_pre | secondary | 5 | -1.1269 | [-2.2534, -0.0005] | descriptive |
| citeseer | 0.1 | pre_minus_post | primary | 5 | 0.3054 | [-0.1826, 0.7934] | 0.75 |
| citeseer | 0.1 | parallel_minus_post | secondary | 5 | -1.8325 | [-3.1283, -0.5368] | descriptive |
| citeseer | 0.1 | parallel_minus_pre | secondary | 5 | -2.1380 | [-3.6613, -0.6146] | descriptive |
| cora | 0.05 | pre_minus_post | primary | 5 | 0.3576 | [-0.1388, 0.8541] | 0.75 |
| cora | 0.05 | parallel_minus_post | secondary | 5 | 0.7331 | [0.0595, 1.4068] | descriptive |
| cora | 0.05 | parallel_minus_pre | secondary | 5 | 0.3755 | [-0.0932, 0.8442] | descriptive |
| cora | 0.1 | pre_minus_post | primary | 5 | -0.1967 | [-0.9379, 0.5445] | 0.75 |
| cora | 0.1 | parallel_minus_post | secondary | 5 | -0.7331 | [-1.5390, 0.0728] | descriptive |
| cora | 0.1 | parallel_minus_pre | secondary | 5 | -0.5364 | [-0.9046, -0.1682] | descriptive |

See summary.json for all missing/rejected stages and paired_differences.csv for each seed's contrasts.
Source/data/environment fingerprints are checked from the saved manifest; historical results do not need to match the current checkout. No manuscript is modified.
