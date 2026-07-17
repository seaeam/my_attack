# GBHA Experiment Suite

This directory contains experiment-only code for the seven empirical requirements in the GBHA manuscript. It deliberately leaves the canonical `meta.py` and `heir.py` paths unchanged.

The theoretical requirement, **GB sensitivity preservation**, is not a runnable experiment. Its empirical counterpart is `coarse_gradient_consistency.py`, which measures whether coarse ball-pair gradients preserve fine node-pair gradient rankings.

## Directory layout

```text
gbhaa_experiments/
├── run_matrix.py                    # matched dataset/seed/budget runner
├── specs.py                         # experiment and variant definitions
├── runner.py                        # timing, peak RSS, output parsing
├── coarse_gradient_consistency.py   # coarse-versus-fine gradient fidelity
├── stealthiness.py                  # structural and feature-space metrics
├── run_stealthiness.py              # artifact generation + analysis
├── aggregate_results.py             # mean/std CSV over seeds
├── entrypoints/                     # separate experimental attack variants
├── launchers/                       # one explicit launcher per experiment
└── tests/
```

Outputs are written under `gbhaa_experiments/results/`. Every executed run keeps:

- the full stdout/stderr log;
- the redacted command;
- wall-clock time and peak resident memory when `/usr/bin/time` is available;
- OS, Python, CPU count, visible CUDA devices, and GPU model/driver when `nvidia-smi` is available;
- parsed clean/edge/feature/combined accuracy;
- dataset, budget, seed, and variant metadata.

## Execution policy

The launchers default to **dry-run** because the full matrices contain many expensive GNN and LLM runs. Inspect the printed commands first, then set `EXECUTE=1`:

```bash
cd /Users/bytedance/Downloads/Github/my_attack
EXECUTE=1 bash gbhaa_experiments/launchers/run_gb_ablation.sh
```

Use the repository's configured Python environment if it differs from the default:

```bash
PYTHON_BIN=/path/to/python EXECUTE=1 \
  bash gbhaa_experiments/launchers/run_efficiency.sh
```

Dataset lists are configurable without editing the launchers:

```bash
DATASETS="citeseer" EXECUTE=1 \
  bash gbhaa_experiments/launchers/run_hybrid_ablation.sh
```

Text experiments default to the existing local Ollama-compatible endpoint. Override `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OLLAMA_MODEL` for another provider. API keys are redacted from result metadata.

All small-graph experiments default to `citeseer`; Cora and Cora-ML are not part of this suite. Citeseer's cached 500-token vocabulary does not match its 3,703 feature columns, so text-related launchers explicitly opt into a 3,703-token `feature_i` fallback vocabulary. These runs are feature-space LLM ablations, not natural-language semantic attacks. The opt-in is recorded in the command and logs; without it, a mismatched cache is rejected.

## 1. Large-scale efficiency

```bash
bash gbhaa_experiments/launchers/run_efficiency.sh
```

Variants:

- `gb`: Granular-Ball hierarchy;
- `kmeans`: matched K-Means hierarchy;
- `node_level`: `--level 1`, which creates singleton candidates and therefore acts as the no-coarsening reference. It runs only on Citeseer.

Recorded evidence includes wall time, peak RSS, edge accuracy, and the exact configuration. Structural efficiency is isolated with `meta_edge_only.py`, so LLM service latency does not hide the search cost. Run end-to-end text cost separately with the Hybrid/LLM suites.

The default scale ladder is Citeseer, Pubmed, and CS. CS requires the repository's PyG dependencies. `ogbn-arxiv` is not a default because the current canonical loader reaches the Planetoid path before its OGB branch; do not report Arxiv results until that loader path is made executable and verified.

## 2. Granular-Ball ablation

```bash
bash gbhaa_experiments/launchers/run_gb_ablation.sh
```

Variants:

- `gb`;
- `kmeans`;
- `random`, a balanced random hierarchy with the same requested cluster count;
- `node_level`, small graphs only.

All variants use the same attack budget, seed, surrogate, hierarchy depth, and inner optimization settings. This isolates the grouping/search backbone instead of changing several attack components at once.

## 3. Hybrid coupling ablation

```bash
bash gbhaa_experiments/launchers/run_hybrid_ablation.sh
```

Variants and primary reported metrics:

| Variant | Entry point | Primary metric |
|---|---|---|
| `edge_only` | `meta_edge_only.py` | edge-only accuracy |
| `feature_only` | `meta_feature_only.py` | feature-only accuracy with adjacency fixed throughout |
| `parallel` | `meta_independent.py` | combined accuracy |
| `serial` | `meta.py` | combined accuracy |

`feature_only` and `parallel` share the independent feature-gradient seed mechanism. The feature-only entry point never searches for or flips an edge: its adjacency stays fixed during every selection and generation round. `parallel` independently selects attributes but also performs the matched structural action. They therefore require separate runs and produce genuinely different artifacts.

## 4. LLM ablation

```bash
bash gbhaa_experiments/launchers/run_llm_ablation.sh
```

Variants:

- `full`;
- `deterministic_no_llm`: feature-aligned deterministic placeholder generation, with no model/API generation call;
- `without_discriminative_words`: `text_cdl_topk=0`;
- `without_keyword_preservation`: `text_budget_per_node=0`;
- `without_similarity_projection`: `text_similarity_min=0`;
- `without_added_word_cap`: effectively removes the cap.

The deterministic control uses the same 3,703-dimensional placeholder vocabulary and candidate-selection process while making zero external model/API calls. Neither it nor the external-LLM variants support semantic-text claims on Citeseer; they measure attacks in the discrete feature space.

## 5. Coarse gradient consistency

```bash
bash gbhaa_experiments/launchers/run_coarse_gradient_consistency.sh
```

For each requested number of clusters, the script:

1. trains the same internal attack model on the original graph;
2. computes the full node-level adjacency gradient;
3. constructs an importance-guided GB partition;
4. computes the coarse graph gradient;
5. maps each coarse ball pair to the best feasible fine node-pair action inside it;
6. reports Spearman/Kendall correlation, action agreement, top-k overlap, top-1 hit, and search regret.

The reference is the **fine first-order gradient**, not an exhaustive nonlinear retraining result after every possible edge flip. State this distinction in the paper. The script is intentionally restricted to small graphs because the fine reference uses a dense node-level adjacency gradient. `--max-nodes` is available only for smoke tests and debugging, not final manuscript evidence.

## 6. AE-PPR comparison

```bash
bash gbhaa_experiments/launchers/run_ae_ppr_comparison.sh
```

This is a clean 2x2 factorial comparison:

| Variant | Global transition | Local endpoint-conditioned transition |
|---|---|---|
| `ppr_ppr` | PPR | PPR |
| `ae_ppr_ppr` | AE-PPR | PPR |
| `ppr_ae_ppr` | PPR | AE-PPR |
| `ae_ppr_ae_ppr` | AE-PPR | AE-PPR |

All local variants use the same two structural endpoints, restart probability, iteration count, and top-k. Only the cosine feature weighting changes. This avoids the confound in the existing `global_pagerank` branch, which also changes the source distribution and search scope.

## 7. Stealthiness analysis

```bash
bash gbhaa_experiments/launchers/run_stealthiness.sh
```

The experiment-only artifact entry point saves original/modified adjacency, features, and labels. The analyzer reports:

Structural metrics:

- edge flip count/rate;
- self-loops and asymmetry;
- degree-distribution JS divergence;
- mean/max degree shift;
- connected component shift;
- homophily shift;
- transitivity shift.

Feature-space metrics:

- modified node rate;
- changed feature entries;
- mean/median/min cosine similarity;
- active-feature Jaccard similarity;
- added/removed features per modified node;
- feature-mean shift.

These are **feature-space** metrics. Text-level semantic similarity, fluency, or BERTScore requires paired original and generated raw text, which the current attack artifact does not retain. Do not rename BoW cosine similarity as natural-language semantic similarity.

For very large graphs, disable sparse triangle multiplication with `--skip-transitivity` and report that omission.

## Aggregating seeds

Each matrix writes an append-only `runs.jsonl`. The aggregator excludes failed commands and keeps only the latest successful record for each dataset/variant/budget/seed, so rerunning a seed does not overweight it. Convert one or more files to a mean/std CSV:

```bash
/opt/miniconda3/bin/python -m gbhaa_experiments.aggregate_results \
  gbhaa_experiments/results/gb_ablation/runs.jsonl \
  --output gbhaa_experiments/results/gb_ablation/summary.csv
```

Do not report a final manuscript table without clean accuracy, explicit budget, seed count, mean, standard deviation, and hardware information.

## Low-cost checks

```bash
/opt/miniconda3/bin/python -m unittest discover \
  -s gbhaa_experiments/tests -v

/opt/miniconda3/bin/python -m compileall -q gbhaa_experiments

bash -n gbhaa_experiments/launchers/*.sh
```
