# GBHAA End-to-End Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the structure-only efficiency matrix with 90 full structure-plus-LLM-attribute attacks on Citeseer and Cora, using each dataset's accepted parameters and real cached vocabulary so GB, K-Means, and node-level total time/memory can be compared fairly.

**Architecture:** Keep `gbhaa_experiments.run_matrix` and its whole-process timing path as the orchestrator. Make all three efficiency variants dispatch the canonical `meta.py` entry point, add an explicit opt-in for a real cached vocabulary shorter than the feature matrix, and run two dataset-specific matrix calls from one launcher. Preserve the existing feature write-back, added-word cap, and similarity projection behavior.

**Tech Stack:** Bash, Python 3, argparse, scikit-learn `CountVectorizer`, PyTorch attack entry points, Ollama's OpenAI-compatible endpoint.

---

## Scope and safety constraints

- Do not run the 90 formal experiment commands.
- Per the user's instruction, do not add, restore, or run unit tests. Use only compile, Bash syntax, and `EXECUTE=0` command inspection.
- Do not stage or commit files. Preserve the existing staged deletions under `gbhaa_experiments/tests/` and all unrelated worktree changes.
- Do not change the feature write-back block in `heir.py`; only pass the new vocabulary-validation flag into `TextAttackGenerator`.
- Do not use `--allow_fallback_vocabulary` for the efficiency experiment and do not synthesize `feature_i` tokens there.

## File map

- Modify `text_attack_generator.py`: validate and log an explicitly allowed smaller real vocabulary.
- Modify `meta.py`: expose `--allow_partial_vocabulary` on the canonical attack CLI.
- Modify `heir.py`: propagate the new CLI flag through both local-Llama and API-backed generator construction paths.
- Modify `gbhaa_experiments/runner.py`: classify the new flag as text-only and record the selected Ollama model in result environment metadata; retain existing timing code.
- Modify `gbhaa_experiments/specs.py`: make efficiency variants complete `meta.py` attacks and allow node-level Cora.
- Modify `gbhaa_experiments/launchers/run_efficiency.sh`: run fixed Citeseer and Cora matrices with their authoritative parameters and external LLM configuration.
- Modify `gbhaa_experiments/README.md`: document the complete efficiency experiment, real partial vocabularies, metric scope, and matrix size.
- Modify `gbhaa_experiments/EXPERIMENT_SUPPLEMENT_PLAN_ZH.md`: update protocol, counts, validation, smoke test placement, run order, and reporting cautions.

### Task 1: Add explicit partial real-vocabulary compatibility

**Files:**
- Modify: `text_attack_generator.py:42-146`
- Modify: `meta.py:128-147`
- Modify: `heir.py:145-190`
- Modify: `gbhaa_experiments/runner.py:22-50,146-160`

- [ ] **Step 1: Extend the generator interface without changing strict defaults**

Add `allow_partial_vocabulary: bool = False` immediately after `allow_fallback_vocabulary` in `TextAttackGenerator.__init__`, document it, and initialize `self.uses_partial_vocabulary = False` beside `self.uses_fallback_vocabulary`.

In the existing-cache branch, keep exact alignment as the first case, then insert this case before fallback handling:

```python
elif allow_partial_vocabulary and feature_dim is not None:
    if 0 < len(self.vocab) < int(feature_dim):
        self.uses_partial_vocabulary = True
        print(
            "Loaded partial real BoW vocabulary: "
            f"{len(self.vocab)} tokens for feature dimension {feature_dim}; "
            "preserving cached word order and leading-column write-back"
        )
    else:
        raise ValueError(
            f"Partial BoW vocabulary must be nonempty and smaller than the "
            f"feature dimension: {len(self.vocab)} vs {feature_dim}"
        )
```

Keep `allow_fallback_vocabulary` as the following branch. This ordering means that when partial mode is explicitly selected, a larger cache is rejected instead of silently converted to placeholders. Update mismatch errors to mention both explicit alternatives while preserving `FileNotFoundError` when the cache itself is absent.

- [ ] **Step 2: Expose the opt-in only through the canonical entry point**

Add this argument beside `--allow_fallback_vocabulary` in `meta.py`:

```python
parser.add_argument(
    "--allow_partial_vocabulary",
    action="store_true",
    default=False,
    help=(
        "Explicitly allow a nonempty cached real vocabulary shorter than the "
        "feature dimension; existing leading-column write-back remains unchanged"
    ),
)
```

Do not add the flag to unrelated `meta_*` entry points.

- [ ] **Step 3: Propagate the flag through both generator construction paths**

In both `TextAttackGenerator(...)` calls in `heir.py`, add:

```python
allow_partial_vocabulary=getattr(
    args, "allow_partial_vocabulary", False
),
```

Do not edit the `new_bow` alignment, added-word limiting, similarity interpolation, or feature assignment around `heir.py:805-862`.

- [ ] **Step 4: Keep shared runner behavior coherent and auditable**

Add `"--allow_partial_vocabulary": 0` to `EDGE_ONLY_UNSUPPORTED_OPTIONS`, because it is a text-only option. Add `"ollama_model": os.environ.get("OLLAMA_MODEL")` to `runtime_environment()` so every saved result records the fixed model identifier. Do not change `_timed_command`, `run_command`, peak-RSS parsing, or wall-clock timing.

- [ ] **Step 5: Perform syntax-level interface checks**

Run:

```bash
"$(command -v python)" -m py_compile \
  text_attack_generator.py meta.py heir.py gbhaa_experiments/runner.py
"$(command -v python)" meta.py --help | rg -- '--allow_partial_vocabulary'
```

Expected: compilation exits 0 and help output contains the new flag. The help command must not start an attack because argparse exits after printing help.

### Task 2: Convert efficiency variants to complete attacks

**Files:**
- Modify: `gbhaa_experiments/specs.py:23-53`

- [ ] **Step 1: Permit both intended small datasets**

Change:

```python
SMALL_DATASETS = frozenset({"citeseer", "cora"})
```

This prevents the existing `small_only=True` gate from skipping Cora's node-level reference.

- [ ] **Step 2: Replace all three structure-only definitions**

Define the efficiency variants as:

```python
"efficiency": (
    Variant(
        "gb",
        "meta.py",
        args=("--coarsen_method", "gb"),
        primary_metric="combined_accuracy",
        needs_text_attack=True,
        uses_external_llm=True,
        description="Complete GB-coarsened structure and LLM-attribute attack.",
    ),
    Variant(
        "kmeans",
        "meta.py",
        args=("--coarsen_method", "kmeans"),
        primary_metric="combined_accuracy",
        needs_text_attack=True,
        uses_external_llm=True,
        description="Complete K-Means-coarsened structure and LLM-attribute attack.",
    ),
    Variant(
        "node_level",
        "meta.py",
        args=("--coarsen_method", "gb", "--level", "1"),
        primary_metric="combined_accuracy",
        needs_text_attack=True,
        uses_external_llm=True,
        small_only=True,
        description="Complete node-level structure and LLM-attribute attack without coarsening.",
    ),
),
```

Do not set `accepts_text_args=False`; the default `True` is required so dataset-specific text parameters reach `meta.py`.

- [ ] **Step 3: Inspect generated variant commands without executing them**

Run a short read-only Python snippet calling `get_variants("efficiency")` and `build_command(...)` for Citeseer and Cora. Assert in the snippet that all targets equal `meta.py`, all commands contain `--use_text_attack`, `--llm_type`, and `--allow_partial_vocabulary`, and none contain `meta_edge_only.py` or `--allow_fallback_vocabulary`.

Expected: the snippet prints `efficiency variant command checks: OK` and exits 0.

### Task 3: Build the fixed two-dataset launcher

**Files:**
- Modify: `gbhaa_experiments/launchers/run_efficiency.sh:1-39`

- [ ] **Step 1: Preserve formal execution defaults and add LLM defaults**

Keep:

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
EXECUTE="${EXECUTE:-1}"
```

Remove the generic `DATASETS` parsing and add:

```bash
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b-instruct-fp16}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"
```

- [ ] **Step 2: Create common and dataset-specific argument arrays**

Use a common array containing `Meta-Both`, `normal`, 200 epochs, step 1, frozen structure features, and `--allow_partial_vocabulary`. Do not include `--coarsen_method`, because the matrix variant appends the matched method.

Create `CITESEER_ARGS` with exactly:

```bash
--level 4 --miter 60 --lr 0.05
--global_important_ratio 0.45 --global_ppr_alpha 0.08
--global_ppr_iters 120 --global_seed_strategy degree
--text_attack_max_visits 4 --text_retries 3 --text_budget_per_node 35
--text_topk_ratio 0.08 --text_ppr_alpha 0.26 --text_ppr_iters 30
--text_min_cluster_size 2 --text_max_cluster_size 5
--text_similarity_min 0.65 --text_cdl_topk 14
--text_cluster_attr_topk 14 --text_max_added_words 40
```

Create `CORA_ARGS` with exactly:

```bash
--level 2 --miter 450 --lr 0.012
--global_important_ratio 0.25 --global_ppr_alpha 0.08
--global_ppr_iters 180 --global_seed_strategy uniform
--text_attack_max_visits 16 --text_retries 3 --text_budget_per_node 80
--text_topk_ratio 0.12 --text_ppr_alpha 0.28 --text_ppr_iters 60
--text_min_cluster_size 2 --text_max_cluster_size 4
--text_similarity_min 0.35 --text_cdl_topk 36
--text_cluster_attr_topk 36 --text_max_added_words 70
```

These values come from `run_citeseer_gb.sh` and `run_cora.sh`; do not replace them with one generic parameter set.

- [ ] **Step 3: Invoke the same matrix function once per dataset**

Define a Bash function accepting a dataset followed by base arguments. It must call:

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_matrix \
  --experiment efficiency \
  --datasets "$dataset" \
  --seeds 15 16 17 18 19 \
  --ptb-rates 0.01 0.05 0.10 \
  --python "$PYTHON_BIN" \
  --llm-type gpt \
  --api-base-url "$OPENAI_BASE_URL" \
  ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
  -- "$@"
```

Call it serially for Citeseer and then Cora using `COMMON_ARGS` plus the matching dataset array. This produces 45 commands per call and 90 total. Do not include Pubmed or CS and do not provide an efficiency `DATASETS` override that can reintroduce them.

- [ ] **Step 4: Check Bash syntax only**

Run:

```bash
bash -n gbhaa_experiments/launchers/run_efficiency.sh
```

Expected: exit 0 with no output.

### Task 4: Update experiment documentation and reporting boundaries

**Files:**
- Modify: `gbhaa_experiments/README.md`
- Modify: `gbhaa_experiments/EXPERIMENT_SUPPLEMENT_PLAN_ZH.md`

- [ ] **Step 1: Rewrite the README efficiency section**

Document:

- fixed datasets Citeseer and Cora; Pubmed and CS are excluded;
- budgets 1%, 5%, 10% and seeds 15-19;
- 90 full runs, with `gb`, `kmeans`, and `node_level` on both datasets;
- all variants execute structure and external-LLM attribute attack through `meta.py`;
- primary effectiveness metric is Combined Accuracy, while Clean/Edge/Feature/Combined Accuracy, drops, perturbation count, LLM calls/cache hits, full-command wall time, and peak RSS are retained;
- GB versus node-level is the main coarsening-savings comparison; GB versus K-Means isolates the type of coarsening;
- dataset-specific parameters follow `run_citeseer_gb.sh` and `run_cora.sh` rather than one shared block.

Replace the claim that all Citeseer text experiments use placeholders with an explicit scope distinction: efficiency uses the 500 real cached words in their original order via `--allow_partial_vocabulary`; other existing launchers that still pass `--allow_fallback_vocabulary` remain feature-space ablations.

- [ ] **Step 2: State measurement scope and formal timing protocol**

In both documents, state that wall time covers the complete attack command and therefore includes waiting for LLM responses. State that `/usr/bin/time` peak RSS covers the attack command, not the separately running Ollama service or its GPU memory.

Require one Ollama warm-up before the formal matrix, the same loaded-model/service state and model identifier for all 90 serial runs, fixed generation settings, and no concurrent experiment jobs. Mention that `OLLAMA_MODEL` is saved in `runtime_environment` for auditability.

- [ ] **Step 3: Correct the full protocol and accounting document**

Update `EXPERIMENT_SUPPLEMENT_PLAN_ZH.md` so that:

- the efficiency dataset row is Citeseer and Cora;
- the efficiency parameter section lists the two authoritative parameter blocks;
- the vocabulary section explains exact leading-column zero-padding and possible restoration by similarity interpolation, without saying the unused columns are always unchanged or always zero;
- the matrix table reads 90 efficiency tasks, all 90 LLM-enabled;
- suite totals read 340 attack tasks plus one gradient diagnostic, with 240 LLM-enabled tasks;
- the no-LLM structure smoke example uses `--experiment gb_ablation`, not `efficiency`;
- the formal run order places efficiency among the Ollama-dependent experiments;
- obsolete instructions that efficiency is structure-only, supports Pubmed/CS, or can be narrowed through its removed `DATASETS` variable are deleted.

- [ ] **Step 4: Keep validation instructions consistent with the user's request**

Remove stale unit-test commands and expectations from these two experiment documents. Retain compile checks, Bash syntax checks, dry-run preview, and result aggregation guidance. Do not alter or restore the deleted test files themselves.

### Task 5: Validate the complete planned matrix without running attacks

**Files:**
- Inspect only: all files modified in Tasks 1-4

- [ ] **Step 1: Compile modified Python modules**

Run:

```bash
"$(command -v python)" -m py_compile \
  text_attack_generator.py meta.py heir.py \
  gbhaa_experiments/specs.py gbhaa_experiments/runner.py \
  gbhaa_experiments/run_matrix.py
```

Expected: exit 0 with no traceback.

- [ ] **Step 2: Check all launcher syntax**

Run:

```bash
bash -n gbhaa_experiments/launchers/*.sh
```

Expected: exit 0 with no output.

- [ ] **Step 3: Preview the efficiency launcher**

Run:

```bash
EXECUTE=0 bash gbhaa_experiments/launchers/run_efficiency.sh \
  > /tmp/gbhaa_efficiency_preview.txt
```

This is command generation only; it must not start GNN training or contact Ollama.

Verify:

```bash
rg -c '^  ' /tmp/gbhaa_efficiency_preview.txt
rg -c 'meta.py' /tmp/gbhaa_efficiency_preview.txt
rg -c -- '--dataset citeseer' /tmp/gbhaa_efficiency_preview.txt
rg -c -- '--dataset cora' /tmp/gbhaa_efficiency_preview.txt
```

Expected counts: 90 indented command lines, 90 `meta.py` occurrences, 45 Citeseer commands, and 45 Cora commands. Also confirm there are two `Dry-run complete: 45 commands planned` summaries.

- [ ] **Step 4: Check required and forbidden arguments**

Inspect the preview and require every command to contain `--use_text_attack`, `--allow_partial_vocabulary`, `--llm_type gpt`, and the API base URL. Require zero occurrences of `meta_edge_only.py`, `--allow_fallback_vocabulary`, `--dataset pubmed`, and `--dataset cs`.

Spot-check one GB, one K-Means, and one node-level command for each dataset. Confirm `node_level` ends with `--level 1`, overriding the dataset base depth while retaining all attribute parameters.

- [ ] **Step 5: Inspect only the scoped diff and worktree state**

Run:

```bash
git diff -- \
  text_attack_generator.py meta.py heir.py \
  gbhaa_experiments/runner.py gbhaa_experiments/specs.py \
  gbhaa_experiments/launchers/run_efficiency.sh \
  gbhaa_experiments/README.md \
  gbhaa_experiments/EXPERIMENT_SUPPLEMENT_PLAN_ZH.md
git status --short
```

Expected: the scoped diff implements only this design; the pre-existing staged test deletions and unrelated launcher/document changes remain untouched. Do not run `git add`, `git commit`, or any formal attack command.
