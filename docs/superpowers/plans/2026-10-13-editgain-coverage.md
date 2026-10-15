# Edit-gain Coverage Implementation Plan

**Goal:** Test whether repetition-aware gain ranking improves Post while preserving matched controls.
**Architecture:** Two opt-in rules and profiles in existing selector/runner, strict audits and a separate launcher. Work directly in the authorized current workspace; preserve all existing user changes.
**Tech Stack:** Python, NumPy/SciPy, PyTorch, unittest, Bash.

- [x] Inspect saved pools/gains/visit histories and record the design.
- [x] Review spec/plan before implementation.
- [x] Add failing coverage unit and launcher tests; run to establish missing behavior.
- [x] Add profiles/preset and deterministic ranking rules; preserve old config/operator identities.
- [x] Extend guided-rule dispatch, version metadata, audit reconstruction, and summary diagnostics/comparisons.
- [x] Add integration/real replay/audit tests covering three arms and multi-round visit history.
- [x] Implement preview/execute/summary launcher and Chinese definitions/commands.
- [x] Run affected regressions, Bash syntax, preview counts and read-only legacy compatibility checks.
- [x] Obtain code review, resolve issues, verify thesis hash and no actual experiment run; deliver script.

Primary files: gbhaa_experiments/post_tuning_protocol.py, selector_mechanisms.py, selector_mechanism_validation.py, entrypoints/meta_post_tuning.py, summarize_post_tuning.py, run_post_tuning.py; new launchers/run_post_coverage.sh and POST_COVERAGE_ZH.md. Tests: new test_post_coverage.py plus relevant selector, entrypoint, summary, runner and action-state regressions.

Verification command: PYTHONPATH=tests /Users/shemingcong/anaconda3/envs/smc/bin/python -m unittest test_post_coverage test_selector_mechanisms test_selector_mechanism_integration test_post_mechanism_launcher test_post_refinement test_post_tuning_protocol test_post_tuning_summary test_post_tuning_entrypoint test_post_tuning_runner test_action_state_protocol test_action_state_replay test_action_state_runner test_action_state_summary

Verification outcome (2026-10-13): 126 tests passed; Bash syntax and 24/12-trial previews passed without creating experiment roots. All 31 legacy profile/operator dictionaries equal the pre-edit snapshot; 82 saved jobs pass catalog identity validation. All 24 saved mechanism results pass full read-only artifact validation. Code reviewer found no blocking or important defect and passed 1,200 in-memory edge cases. No real attacks, model fitting, or LLM calls were run. Thesis SHA-256 remains `4a41345f0515644f5f4737f35daa4a62729affebfbb667552eb9f6549e2a494e`.
