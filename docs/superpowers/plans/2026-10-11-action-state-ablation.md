# Matched action-state ablation implementation plan

**Goal:** Provide an isolated, reproducible Parallel/Endpoint-Pre/Endpoint-Post
experiment and paired analysis that the user can execute.

**Architecture:** Keep the old loop/protocols unchanged. A new entrypoint uses
the existing bootstrap/evaluation code, records a structure-only trajectory,
then implements an explicit replay loop for each arm. Small protocol, runner,
artifact validation, cache and summary modules separate the responsibilities.

**Tech stack:** Python, NumPy/SciPy, PyTorch/DeepRobust, existing OpenAI-compatible
generation, unittest, Bash launchers.

## Tasks

- [x] Add failing synthetic tests for protocol settings, independent restart,
  pre/post selection, exact flip replay and resource counts.
- [x] Implement `gbhaa_experiments/action_state_protocol.py` and
  `gbhaa_experiments/action_state_cache.py`; reuse the existing diffusion and
  projection code. Test content-keyed cache with offline replies.
- [x] Implement `gbhaa_experiments/entrypoints/meta_action_state.py`: planner
  and replay attack, graph-state diagnostics, input/trajectory checks, fixed
  node-update workload, artifacts and inherited victim recording.
- [x] Implement `gbhaa_experiments/action_state_validation.py`: validate
  trajectory replay, actual graph/features, predictions, budgets and identity.
- [x] Implement `gbhaa_experiments/run_action_state.py` and
  `gbhaa_experiments/launchers/run_action_state.sh`: plan-only default, shared
  plans, isolated results, exact-fingerprint resume and archived failed retry.
- [x] Implement `gbhaa_experiments/summarize_action_state.py`: complete blocks,
  per-seed ACC/differences, mean/sample SD/95% CI and actual perturbation metrics.
- [x] Add runnable guidance `gbhaa_experiments/ACTION_STATE_EXPERIMENT_ZH.md`.
- [x] Run offline unit/integration tests and existing relevant regressions;
  inspect commands for 20 plans/60 arms, smoke plans and optional Pubmed.
- [x] Review final implementation for scientific matching and artifact integrity,
  fix material issues, verify thesis/old-loop checksums, give commands to user.

All modifications stay in the user's working directory with existing edits
preserved. No commits, real attacks, API calls or thesis changes are required.

## Verification completed

- 84 offline tests passed, including actual synthetic planner/replay artifacts
  through validation and paired summary, new cache-integrity tests, and relevant
  existing source/selector/factorial/recording/generation/budget regressions.
- Bash syntax, default 20-plan/60-arm dry run, single-block dry run and optional
  Pubmed dry run passed. No real experiment output directory was created.
- Original main-loop AST source and `Main-AAAI.tex` checksums match the pre-edit
  snapshot. Independent implementation review approved after cache/input audit
  fixes. No real attacks or LLM requests were executed.
