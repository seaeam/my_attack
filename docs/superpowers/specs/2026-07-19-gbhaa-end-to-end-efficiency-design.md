# GBHAA End-to-End Efficiency Experiment Design

## Goal

Change the first supplementary experiment from a structure-only benchmark into an end-to-end GBHAA efficiency comparison. The experiment must measure total wall-clock time and peak process memory for the complete structure-plus-LLM-attribute attack, so the results can evaluate whether GB coarsening reduces the resource cost of the full method without hiding an unacceptable loss in attack effectiveness.

## Authoritative Configurations

Dataset-specific parameters come from the repository's accepted launchers:

- Citeseer: `run_citeseer_gb.sh`
- Cora: `run_cora.sh`

The efficiency experiment uses `meta.py` for both datasets so each coarsening variant runs through the same canonical attack and evaluation entry point. Only parameters from the launchers are dataset-specific; dataset, seed, perturbation rate, and coarsening variant remain matrix-controlled fields.

### Citeseer parameters

- `model=Meta-Both`
- `level=4`, `step=1`, `miter=60`, `lr=0.05`
- global PPR: ratio `0.45`, alpha `0.08`, iterations `120`, seed strategy `degree`
- frozen structure-search features
- text attack: max visits `4`, retries `3`, per-node budget `35`
- local text PPR: ratio `0.08`, alpha `0.26`, iterations `30`
- text clusters: size `2` to `5`
- similarity minimum `0.65`
- discriminative and cluster attribute words: `14` each
- maximum added words `40`

### Cora parameters

- `model=Meta-Both`
- `level=2`, `step=1`, `miter=450`, `lr=0.012`
- global PPR: ratio `0.25`, alpha `0.08`, iterations `180`, seed strategy `uniform`
- frozen structure-search features
- text attack: max visits `16`, retries `3`, per-node budget `80`
- local text PPR: ratio `0.12`, alpha `0.28`, iterations `60`
- text clusters: size `2` to `4`
- similarity minimum `0.35`
- discriminative and cluster attribute words: `36` each
- maximum added words `70`

Both datasets use the launchers' existing Ollama-compatible GPT configuration.

## Experiment Matrix

- Datasets: `citeseer`, `cora`; Pubmed and CS are excluded.
- Perturbation rates: `0.01`, `0.05`, `0.10`.
- Seeds: `15`, `16`, `17`, `18`, `19`.
- Variants:
  - `gb`: complete attack with Granular-Ball hierarchical search.
  - `kmeans`: complete attack with matched K-Means hierarchical search.
  - `node_level`: complete attack with `level=1`, providing the no-coarsening reference.
- Total: 2 datasets × 3 budgets × 5 seeds × 3 variants = 90 complete attack runs.

All variants enable text attack and external LLM calls. Their primary effectiveness metric is Combined Accuracy. The runner continues to capture Clean, Edge, Feature, and Combined Accuracy, accuracy drops, LLM calls, cache hits, completed structure perturbations, total wall time, and peak RSS.

The wall-clock measurement covers the entire attack command, including time spent waiting for external LLM responses. Peak RSS is the value reported for the attack command by `/usr/bin/time`; it does not include the separately running Ollama server process or its GPU memory. Results and documentation must state this scope explicitly rather than presenting peak RSS as total host or LLM-service memory.

The primary efficiency comparison is `gb` versus `node_level`. `gb` versus `kmeans` distinguishes GB-specific behavior from generic clustering-based coarsening. Results are compared within the same dataset, budget, and seed; the two datasets' different canonical hyperparameters are not compared as if they were a single shared configuration.

## Partial Real Vocabulary Compatibility

The authoritative scripts and historical logs use the existing 500-word `bow_cache/citeseer.pkl` and `bow_cache/cora.pkl` vocabularies with feature matrices of 3,703 and 1,433 columns respectively. Preserve the existing `heir.py` write-back behavior exactly: place the generated 500-dimensional BoW vector in the leading 500 feature columns and zero-pad it to the full feature dimension, then apply the existing added-word cap. If the resulting vector falls below the dataset's cosine-similarity threshold, the existing linear projection interpolates it with the original full-dimensional feature vector. Consequently, trailing columns are zero immediately after padding but may be restored proportionally by that projection. This experiment changes vocabulary validation only; it does not change feature write-back, added-word limiting, or similarity projection.

Do not pass `--allow_fallback_vocabulary` and do not replace the words with `feature_i` placeholders.

Add an explicit `--allow_partial_vocabulary` option to the canonical `meta.py` path. When selected:

- an existing nonempty vocabulary smaller than `feature_dim` is accepted;
- the original cached real words and their existing order are preserved;
- the run clearly logs the partial vocabulary size and full feature dimension;
- a vocabulary larger than the feature dimension remains invalid;
- no placeholder tokens are generated;
- strict exact-dimension validation remains the default for every other experiment.

The efficiency launcher explicitly opts into this compatibility mode, making the historical assumption visible in saved commands and logs rather than silently weakening validation globally.

## Launcher Structure

`run_efficiency.sh` invokes the existing matrix runner twice, once per dataset, because Citeseer and Cora require different base argument arrays. Both invocations write to the same `results/efficiency` hierarchy and run serially. The launcher:

- defaults to formal execution, retaining `EXECUTE=0` preview mode;
- resolves Python from the active environment;
- exports the same Ollama model, key placeholder, and base URL convention as the other full text-attack launchers;
- passes `--llm-type gpt` and the configured API base URL;
- does not include Pubmed or CS.

For formal timing, warm the configured Ollama model once before starting the matrix, keep the same server and loaded-model state for all 90 serial runs, and do not run other experiment jobs concurrently. Use one fixed Ollama model and the same generation configuration/defaults for every variant; do not vary decoding settings between GB, K-Means, and node-level runs. Record the Ollama model identifier and service configuration alongside the result set so the timing protocol is reproducible.

## Documentation and Matrix Accounting

Update `gbhaa_experiments/README.md` and `EXPERIMENT_SUPPLEMENT_PLAN_ZH.md` to describe the experiment as end-to-end rather than structure-only. Document dataset-specific configurations and the partial real vocabulary scope.

The efficiency row changes from 105 non-LLM attacks to 90 LLM-enabled complete attacks. The full suite therefore changes from 355 to 340 attack tasks, while LLM-enabled tasks change from 150 to 240, assuming the other six experiment groups remain unchanged.

Move the existing no-LLM structure smoke-test example from `efficiency` to `gb_ablation`, because every efficiency variant now requires the full attribute path.

## Validation

Per the user's earlier instruction, do not add or run unit tests and do not start a formal experiment. Validate with:

- Python compile checks for modified Python modules;
- Bash syntax checks;
- `EXECUTE=0` command preview confirming exactly 90 planned commands;
- command inspection confirming both datasets use `meta.py`, text attack, external LLM arguments, partial real vocabulary mode, and their own canonical parameters;
- checks confirming no efficiency command uses `meta_edge_only.py`, `--allow_fallback_vocabulary`, Pubmed, or CS;
- final scoped diff inspection that preserves unrelated staged deletions and changes.
