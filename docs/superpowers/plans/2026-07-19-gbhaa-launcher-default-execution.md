# GBHAA Launcher Default Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every GBHAA launcher use the active Python environment and execute formal experiments by default, while keeping `EXECUTE=0` as an explicit preview override.

**Architecture:** Keep the existing launcher-to-runner flow unchanged. Only change launcher defaults and synchronize the two experiment documents; do not modify experiment matrices or low-level Python runner interfaces.

**Tech Stack:** Bash, Markdown, existing Python experiment runners.

---

### Task 1: Update all launcher defaults

**Files:**

- Modify: `gbhaa_experiments/launchers/run_ae_ppr_comparison.sh:7-8`
- Modify: `gbhaa_experiments/launchers/run_coarse_gradient_consistency.sh:7-8`
- Modify: `gbhaa_experiments/launchers/run_efficiency.sh:7-8`
- Modify: `gbhaa_experiments/launchers/run_gb_ablation.sh:7-8`
- Modify: `gbhaa_experiments/launchers/run_hybrid_ablation.sh:7-8`
- Modify: `gbhaa_experiments/launchers/run_llm_ablation.sh:7-8`
- Modify: `gbhaa_experiments/launchers/run_stealthiness.sh:7-8`

- [ ] **Step 1: Replace the hard-coded interpreter default in every launcher**

Use the current environment's `python`, while retaining explicit `PYTHON_BIN` overrides:

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
```

- [ ] **Step 2: Make formal execution the default in every launcher**

```bash
EXECUTE="${EXECUTE:-1}"
```

Do not change the existing conditionals: with the new default they add `--execute` or directly run the gradient diagnostic; `EXECUTE=0` continues to select the existing preview path.

- [ ] **Step 3: Inspect the launcher-only diff**

Run:

```bash
git diff -- gbhaa_experiments/launchers
```

Expected: only the `PYTHON_BIN` and `EXECUTE` default lines change.

### Task 2: Synchronize user-facing commands and behavior descriptions

**Files:**

- Modify: `gbhaa_experiments/README.md`
- Modify: `gbhaa_experiments/EXPERIMENT_SUPPLEMENT_PLAN_ZH.md`

- [ ] **Step 1: Update README execution instructions**

State that launchers formally execute by default. Show a direct `bash ...` command as the normal path, document `EXECUTE=0` as optional preview mode, remove redundant `EXECUTE=1`, replace `/opt/miniconda3/bin/python` examples with `python` or `PYTHON_BIN="$(command -v python)"`, and remove the machine-specific `/Users/bytedance/Downloads/Github/my_attack` repository path.

- [ ] **Step 2: Update the full experiment plan**

Change environment setup to:

```bash
export PYTHON_BIN="$(command -v python)"
```

Change the command-preview section to explicitly prefix launcher calls with `EXECUTE=0`. Remove `EXECUTE=1` from the formal run order and dataset-filter examples. Preserve `--execute` in direct `python -m gbhaa_experiments.run_matrix` smoke-test commands because the low-level runner interface is intentionally unchanged.

- [ ] **Step 3: Inspect the documentation diff**

Run:

```bash
git diff -- gbhaa_experiments/README.md gbhaa_experiments/EXPERIMENT_SUPPLEMENT_PLAN_ZH.md
```

Expected: command paths and default-execution descriptions change; scientific settings remain untouched.

### Task 3: Perform non-executing static verification

**Files:**

- Verify: `gbhaa_experiments/launchers/*.sh`
- Verify: `gbhaa_experiments/README.md`
- Verify: `gbhaa_experiments/EXPERIMENT_SUPPLEMENT_PLAN_ZH.md`

Per user instruction, do not add or run unit tests and do not launch any formal experiment.

- [ ] **Step 1: Validate Bash syntax**

Run:

```bash
bash -n gbhaa_experiments/launchers/*.sh
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Check for stale launcher defaults and hard-coded interpreter paths**

Run:

```bash
rg -n '/opt/miniconda3/bin/python|/Users/bytedance/Downloads/Github/my_attack|EXECUTE="\$\{EXECUTE:-0\}"' gbhaa_experiments
```

Expected: no matches.

- [ ] **Step 3: Confirm every launcher has the intended defaults**

Run:

```bash
rg -n 'PYTHON_BIN="\$\{PYTHON_BIN:-\$\(command -v python\)\}"|EXECUTE="\$\{EXECUTE:-1\}"' gbhaa_experiments/launchers
```

Expected: 14 matches total, two in each of seven launchers.

- [ ] **Step 4: Check for outdated default dry-run wording**

Run:

```bash
rg -n '默认只执行.*dry-run|默认不会真正训练|Commands are dry-run unless' \
  gbhaa_experiments/README.md \
  gbhaa_experiments/EXPERIMENT_SUPPLEMENT_PLAN_ZH.md
```

Expected: no matches. Both documents must describe preview as an explicit `EXECUTE=0` override.

- [ ] **Step 5: Review the complete scoped diff**

Run:

```bash
git diff --check
git diff -- \
  gbhaa_experiments/launchers \
  gbhaa_experiments/README.md \
  gbhaa_experiments/EXPERIMENT_SUPPLEMENT_PLAN_ZH.md \
  docs/superpowers/specs/2026-07-19-gbhaa-launcher-default-execution-design.md \
  docs/superpowers/plans/2026-07-19-gbhaa-launcher-default-execution.md
```

Expected: no whitespace errors and no changes to experiment parameters, matrices, or Python execution logic.
