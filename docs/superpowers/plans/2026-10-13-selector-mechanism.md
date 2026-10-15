# Selector Mechanism Implementation Plan

**Goal:** Provide a bounded, matched experiment testing endpoint budget allocation and surrogate-guided feature vulnerability, with unchanged generation and no thesis edits.

**Architecture:** Add an opt-in mechanism module and three profile identities; minimally extend the existing tuning entrypoint and validator. Reuse the runner/evaluation/summary. Keep all legacy profile dictionaries and result matrices unchanged.

**Tech Stack:** NumPy/SciPy, PyTorch, unittest, Bash.

### Task 1: Diagnostic evidence

- [x] Read saved updates and rounds for all 16 confirmation trials; save descriptive mechanism diagnostics.
- [x] Inspect selection, grouping, projection and gradient code; distinguish observed evidence from hypotheses.
- [x] Save a concise Chinese diagnosis and method definitions.

### Task 2: Algorithm and profile tests first

- [x] Add failing tests for frontier support/fallback, visits, bilateral ordering, gradient/gain and profile matrices.
- [x] Add `selector_mechanisms.py`; extend catalog/schema/operator metadata for only new profiles and expose mechanism preset.
- [x] Run targeted unit tests, preserving old profiles and presets.

### Task 3: Replay integration and audit

- [x] Add failing three-arm dispatch, pseudo-label and tampered-trace tests.
- [x] Add per-graph dispatch and trace recording in `meta_post_tuning.py`; use the existing fixed surrogate without fitting.
- [x] Extend `post_tuning_validation.py` to audit new traces and pseudo-label artifacts without changing old artifact requirements.
- [x] Run actual synthetic replay and legacy entrypoint/validator regression tests.

### Task 4: Launcher and verification

- [x] Add preview/summary launcher tests; implement `run_post_mechanism.sh` and Chinese instructions.
- [x] Run mechanism and affected legacy tests; check Bash syntax and actual preview commands (24 jobs default, 12 single-seed).
- [x] Compare all 28 old profile identities and validate saved old results without overwriting them.
- [x] Obtain focused code review, resolve findings, verify thesis hash and no experiment execution; provide commands.

## Verification outcome

- Independent review approved after repairing a one-ULP score/tie-break audit issue; near-equal saved scores cannot change the authoritative recomputed ranking.
- Tests exercise all three new rules across all three arms, real synthetic replay/load_trial, fixed pseudo-labels, the actual DeepRobust GCN forward/gradient, and preview-only launchers.
- Read-only validation accepted all 58 saved tuning/refinement/confirmation results; their saved catalog identities are unchanged.
- No training/LLM experiments were launched; the new experiment output root does not exist. Thesis SHA-256 remains 4a41345f0515644f5f4737f35daa4a62729affebfbb667552eb9f6549e2a494e.
