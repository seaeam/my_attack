# Endpoint-Post Quick Tuning Implementation Plan

> Use subagent-driven development with independent scoped implementation and review.

**Goal:** Deliver a small user-run parameter/method sweep for Endpoint-Post.

**Architecture:** New versioned protocol and isolated files reuse the existing
bootstrap and saved-action replay. Validation-only evaluation and strict artifact
checks feed a complete-block ranking; old experiments remain untouched.

**Tech Stack:** Python, NumPy/SciPy/PyTorch, DeepRobust, unittest, Bash.

## Tasks

- [x] Review spec and identify data-flow/integrity issues before implementation.
- [x] Add profile catalog/scorer/job builder and meaningful synthetic tests.
- [x] Add subclass entrypoint, prefix compatibility checks, validation-only victim
  evaluation and tests; reuse legacy replay without editing its implementation.
- [x] Add runner, strict validator and development ranking with completeness,
  fingerprint, split/cache/graph/feature checks and tests.
- [x] Add Bash launcher and Chinese commands/explanations for quick, tiny,
  extended and matched follow-up comparisons.
- [x] Run offline integration tests, original relevant regressions, dry runs,
  independent code review; verify original sources/results/manuscript unchanged.
- [x] Report tested implementation and commands; make no performance claims
  until the user runs the sweep.

No commits, staging, resets, real dataset attacks, or LLM requests.

## Verification outcome

- 156 tests passed across the four new test modules and related action-state,
  selector/factorial, recording, runner, source, and attribute-budget regressions.
- Separately, the legacy `test_followup_protocol` module has four failures among
  thirteen tests: they require a post-flip original loop / removed helper. The
  user explicitly retained that original loop; neither old implementation nor
  tests were changed to hide these failures.
- Shell syntax, actual launcher dry-run (12), and tiny/quick/extended (6/12/22)
  command counts passed; no real output root was created.
- Existing Citeseer .1 parent trajectories for seeds 15/16 validated (366 actions
  each), so both default screening and documented small recheck have a source.
- Baseline hashes for heir.py, meta.py, original action-state files, and thesis
  remain identical. Independent implementation review approved the final path.
