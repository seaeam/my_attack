# Post Refinement Implementation Plan

> **For agentic workers:** Use the existing authorized inline workflow. Review the spec and plan before implementation; keep all edits in the current user workspace and preserve existing staged/untracked work.

**Goal:** Prepare a focused, matched three-arm refinement and multi-seed verification workflow without claiming a predetermined outcome.

**Architecture:** Reuse the v1 scorer/runner/validator. Add catalog profiles, a matched launcher and paired uncertainty reporting; preserve legacy profile definitions and matrices. Store new experimental results separately.

**Tech Stack:** Python, NumPy/SciPy, unittest, Bash.

---

### Task 1: Focused profiles

- [x] Add failing tests to `tests/test_post_tuning_protocol.py` for the ten-profile refine preset, 30 matched jobs, unchanged generation and original presets.
- [x] Verify targeted tests fail because profiles are absent.
- [x] Add eight profiles in `gbhaa_experiments/post_tuning_protocol.py`, preserving the original twenty-profile extended preset; expose refine in `run_post_tuning.py`.
- [x] Check all legacy configs against stored old result metadata; run protocol and entrypoint tests.

### Task 2: Paired reporting

- [x] Add failing summary tests for one-seed undefined CI, known paired multi-seed CI, and ACC ties despite NLL improvement.
- [x] Extend `_compare` in `summarize_post_tuning.py` with ACC-only counts and paired CI; retain selection criterion and legacy fields.
- [x] Add Markdown comparisons and `paired_comparisons.csv`; test CLI report export and missing-result behavior.

### Task 3: Launcher and documentation

- [x] Add failing tests for default screen, confirm recommendation/seeds/prefix and both summary modes, using a fake Python executable without running experiments.
- [x] Add `gbhaa_experiments/launchers/run_post_refinement.sh`, with distinct output roots and default preview behavior.
- [x] Document profiles, budgets, explicit method-variant status of floors/readout, preserved arm meanings, commands and interpretation in `POST_REFINEMENT_ZH.md`.

### Task 4: Verification

- [x] Run all post-tuning protocol/runner/entrypoint/summary tests and relevant action-state regressions.
- [x] Check Bash syntax and preview the real screen command; validate confirmation routing with a temporary recommendation.
- [x] Revalidate existing results into a temporary report directory and compare previous metrics; verify thesis hash unchanged.
- [x] Update this checklist and provide executable commands. No training, LLM calls, commits or thesis changes.

## Completion evidence

- 97 relevant offline tests passed; new regressions were first observed failing.
- Code review approved after fixing rejected/partial-cell interval reporting.
- Existing 12 results revalidated, original configs/rankings/differences retained.
- Screen and confirmation command previews verified (30 × 60, 16 × 120); no experiment started.
- Thesis hash unchanged; no new result directory created.
