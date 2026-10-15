# Projected Editing Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. The user already authorized implementation. Preserve existing workspace and do not commit unrelated changes.

**Goal:** Add two independently ablatable editor mechanisms with matched three-arm quick runs and reproducible audits.

**Architecture:** Isolate pure editing helpers, runtime adapter and saved-artifact auditor. Extend existing tuning catalog/summary conditionally; all legacy profiles retain their identities and writer.

**Tech Stack:** Python, NumPy, PyTorch, scipy, sklearn vectorizer, unittest, bash.

## Task 1: Pure editing and catalog
- [x] Save current catalog/operators and protected file hashes to /tmp/gbca_editing_before.json.
- [x] Add tests/test_projected_editing.py for 12-cell matrix, signed word plans, projection changing candidate ordering, deletion enforcement, duplicates and deterministic ties; run to confirm failure.
- [x] Create gbhaa_experiments/projected_editing.py: word plans, deterministic text adaptation, legacy-compatible addition cap/cosine projection, sequential candidate selection.
- [x] Extend post_tuning_protocol.py with four profiles, editing preset and reference_profile helper. Enforce complete schemas and boolean factor flags.
- [x] Run new unit tests; require pass and unchanged old catalog/operators.

## Task 2: Runtime and audit
- [x] Add tests/test_projected_editing_integration.py using an unfitted tiny real GCN and fake text responses for all 12 cells. Test graph routing, four writes, three requests, gradients and tamper rejection.
- [x] Create projected_editing_runtime.py to snapshot fixed model/vectorizer and execute candidates; dispatch only new profiles in entrypoints/meta_post_tuning.py. Preserve heir.py and legacy writer.
- [x] Create projected_editing_validation.py to reconstruct word plans, vectors, losses and choices from artifacts and checkpoint. Connect post_tuning_validation.py and generation-cache linkage.
- [x] Run integration tests. No model.fit or external generation calls.

## Task 3: Reports and launcher
- [x] Add tests for factorial effects/interaction and incomplete cells; add launcher preview tests.
- [x] Extend summarize_post_tuning.py with new reference, per-arm factorial paired effects and candidate diagnostics. Keep historical report semantics.
- [x] Add run_post_editing.sh with preview, screen, summary modes; editing preset parser; document exact commands and comparison limits in FOLLOWUP_EXPERIMENTS_ZH.md.
- [x] Run all relevant regression suites, bash -n and 12/24-trial preview. Verify protected hashes and old profile identities.
- [x] Request code review using requesting-code-review skill; resolve material issues and rerun affected checks.
- [x] Give user exact preview/run/summary commands, default scope and honest remaining experimental uncertainty.

Review clarification: follow the spec's round-frozen per-node plans and shared three templates; baseline AE-PR is re-evaluated independently per cell. Include nonzero sequential-write, duplicate/all-negative and incomplete-factorial tests. Positive factorial contrasts use control minus treatment ACC; interaction=gain+words-control-both.
