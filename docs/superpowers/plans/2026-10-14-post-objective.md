# ReLU editor objective implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, with test-driven changes and final code review. User already authorized implementation; no extra execution approval or commits.

**Goal:** Deliver a small matched 2×2 objective ablation and runnable launcher based on `transfer_relu`.

**Architecture:** Isolate scalar objectives and frozen target weights in `editor_objectives.py`. Opt in through three new profiles, preserve the old execution path, and extend independent replay auditing and summaries.

**Tech Stack:** Python, NumPy/SciPy/PyTorch, existing unittest artifacts, Bash launcher.

---

### 1. Tests and protocol

Files: `tests/test_post_objective.py`, `tests/test_post_tuning_protocol.py`, `gbhaa_experiments/post_tuning_protocol.py`, `gbhaa_experiments/run_post_tuning.py`.

- [x] Add failing tests for the 2×2 catalog, 24-trial default, explicit override, schema rejection, unchanged old anchor behavior.
- [x] Add analytic capped-margin and weighted-NLL tests, plateau gradient, label-free round weights, uniform fallback, invalid targets/restarts.
- [x] Run `PYTHONPATH=tests /Users/shemingcong/anaconda3/envs/smc/bin/python -m unittest test_post_objective` and record expected import failure.
- [x] Implement three profiles with `editor_objective` in `nll/capped_margin` and `editor_local_mix` in `[0,1)`, requiring ReLU/full replacement; add preset/anchor and operator descriptions.

### 2. Runtime and audit

Files: new `gbhaa_experiments/editor_objectives.py`; modify `projected_editing_runtime.py`, `projected_editing_validation.py`.

- [x] Implement `target_weights(features, graph, restarts, targets, config)` and `objective_value(output,pseudo,targets,weights,kind)`. Validate finite positive normalized weights and target IDs; exclude training nodes by consuming only the existing target set.
- [x] Add optional keyword-only `restarts` to `edit_round`. New profiles freeze weights once, compute the matching scalar gradient, and use it for each actual projected sequential candidate score. Legacy path remains unchanged.
- [x] Runtime passes actual selection restarts only for new profiles; audit independently derives both the expected graph and restarts from original graph/actions/arm/seed rather than trace, then recomputes weights, gradients, candidate bank and chosen writes.
- [x] Trace and verify objective version/type, local mix, restart IDs, target IDs/weights/hash, fallback, diffusion mass and effective target count. Add new-profile/arm replay and wrong-graph/restart/weight tamper tests, gradient/score consistency and freeze checks. No network or real datasets in tests.

### 3. Reports and launcher

Files: `post_editing_summary.py`, `summarize_post_tuning.py`, `tests/test_post_editing_end_to_end.py`; new `launchers/run_post_objective.sh`, `POST_OBJECTIVE_ZH.md`.

- [x] Add objective factorial and per-profile joint advantage rows. Preserve old transfer/editor summaries; no spurious old factorial for the new preset.
- [x] Extend tiny fixture with new IDs; test public 24-cell complete matrix, missing/rejected suppression, correct paired differences and strict-win tie handling.
- [x] Implement launcher default two paired seeds, explicit one-seed override, separate result/report roots, preview/execute/summary/retry usage.
- [x] Document the formulas, expected trial counts and exploratory interpretation in Chinese.

### 4. Verification and review

- [x] Run targeted tests, then relevant `test_post_*.py`, `test_projected_editing*.py`, `test_action_state*.py`, `test_selector_*.py` suite once; fix and rerun on failure.
- [x] Check all old 40 catalog/operator/runtime values against `/tmp/gbca_objective_before.json` and protected hashes; replay representative historical results.
- [x] Request code review under requesting-code-review skill; resolve material findings.
- [x] Save QA evidence, mark steps complete, give exact preview/run/summary commands. Do not claim improved ACC before user runs the experiments.

## Completion evidence

170 offline tests passed; six historical full editor audits passed; all 40 old profile/operator/runtime identities and protected hashes unchanged. Both spec/plan and code reviews approved. Default source prefixes for seeds 15 and 16 are available. Evidence: `gbhaa_experiments/followup_reports/post_objective_implementation_qa.json`. No real experiment execution or thesis edits.
