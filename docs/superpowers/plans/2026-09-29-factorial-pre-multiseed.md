# Factorial Pre-Graph Multi-Seed Implementation Plan

> **For agentic workers:** Execute the approved plan task-by-task with TDD and verification-before-completion. The user has requested implementation in this working project and will run the experiments. No further execution approval, commits, full attacks, generation requests or manuscript changes are required.

**Goal:** Make the existing global/local attribute-weighting 2×2 runnable and auditable on the current pre-flip loop, using paired seeds 15–19 and complete paired global/local/interaction contrasts.

**Architecture:** Add `aepr-factorial-v2-pre-stable` and a separate default result root `followup_results_factorial_pre`. Keep global degree-source `P @ x` accumulation (.15, 30) and endpoint-local `P.T @ s` final-state diffusion (.28, 60). Add a base ranking hook with the exact legacy argsort behavior; override it only for the new factorial protocol with descending scores and ascending node IDs for exact ties. Preserve selector and other legacy protocols. Keep frozen structural features and validate action equality within, never across, each global condition.

**Tech Stack:** Python, NumPy, SciPy, PyTorch, unittest; existing runner, artifact validator and plotting tools.

## Approved scope and files

- `heir.py`: connect `rank_global_candidates` at the existing sorting location; the base implementation returns `np.argsort(-scores)[:k]`. Do not move attribute selection or edge flips.
- `gbhaa_experiments/followup_config.py`: factorial identity and a serializable declaration of source/propagation/initialization/readout/tie/frozen-feature rules.
- `gbhaa_experiments/run_followup.py`: factorial pre jobs and separate default output; reuse existing suite-aware protocol propagation and resume checks.
- `gbhaa_experiments/entrypoints/meta_followup.py`: reject incompatible new-protocol options; stable factorial ranking and per-round diagnostics (global unique-score count, selected nodes, boundary tie count); declare the fixed operator rules in attack artifacts.
- `gbhaa_experiments/summarize_followup.py`: validate operator declarations; add `global_gain_with_topology_local = topology_ppr - attribute_ppr` and `global_gain_with_attribute_local = topology_ae_pr - attribute_ae_pr`; retain both local gains and interaction. Positive contrasts mean lower attacked accuracy under attribute weighting. Existing planned-family Holm correction includes all five contrasts.
- `tests/test_followup_factorial.py` and existing follow-up fixtures: cover numerical operator invariants, protocol isolation, actual ranking dispatch, diagnostics, artifact consistency/resume, direct global/interaction statistics and trajectory rules.
- `gbhaa_experiments/FOLLOWUP_EXPERIMENTS_ZH.md`: document pre/frozen-feature interpretation, separate outputs, no historical relabeling, exact pilot/full/summary/plot commands.

## Tasks

- [x] Write regression tests, run them and confirm expected failures before changing implementation. Preserve the four pre-existing deferred edge-first failures.
- [x] Implement new protocol/defaults/operator metadata and base/subclass ranking dispatch. Test legacy and selector ranking behavior remains exactly argsort.
- [x] Implement all five paired contrasts and fixture validation, including rejection of old protocols, post graphs and missing/mismatched declarations.
- [x] Run a synthetic real-loop four-arm integration without training or API calls; validate protocol/diagnostics/evaluations and same-global action checks.
- [x] Update author instructions; dry-run the real launcher for one seed and both budgets. Run synthetic two-seed summary and plotting in a temporary directory.
- [x] Run follow-up, attribute-budget and streaming regressions; inspect failures; compile changed Python; check changed-file whitespace; verify historical-result hashes unchanged and have a read-only review.

## Validation commands

Use `/Users/shemingcong/anaconda3/envs/smc/bin/python -B -m unittest discover -s tests -p 'test_followup_*.py' -v` for the follow-up suite and patterns `test_attribute_node_budget.py`, `test_runner_streaming.py` for focused regression. Four known tests expect an edge-first loop and are intentionally deferred by the user; new failures must be resolved.

Dry run: `PYTHON_BIN=/Users/shemingcong/anaconda3/envs/smc/bin/python bash gbhaa_experiments/launchers/run_followup.sh factorial --seeds 15 --budgets 0.05 0.1`. It must print eight commands with pre, degree, the new protocol/root and endpoint PPR/AE-PR variants, without creating results.

Full author command (do not run during implementation): `bash gbhaa_experiments/launchers/run_followup.sh factorial --seeds 15 16 17 18 19 --budgets 0.05 0.1 --execute` (40 runs). Summary and figures use explicit `followup_reports/factorial_pre` paths, with exactly the same seeds and budgets.

Existing working-tree changes and historical artifacts belong to the user. Keep this scoped update in the current checkout so the provided commands run the edited code; do not reset, stage, commit or move those changes.

## Verified results

- Follow-up suite: 89 tests, 85 passed; the four existing edge-first failures remain (three missing `prepare_attribute_adjacency`, one expecting the flip before selection). No new failures.
- Focused factorial tests: 9 passed, including the analytic distinct global/local readouts, real-loop ranking dispatch, runtime fixed-rule rejection, protocol/resume isolation and exact legacy ordering with interleaved ties.
- Attribute-budget tests: 4 passed; streaming tests: 6 passed.
- Two-seed/two-budget synthetic reports accepted four complete blocks and ten paired contrasts; both factorial PNG/SVG plots rendered.
- Real launcher dry-run verified eight commands using pre, degree, the new protocol and separate root. Syntax and scoped whitespace checks passed.
- SHA-256 checks confirmed all 717 existing result/artifact/log files under followup result roots unchanged. The new factorial output root was not created: no full attack, training run or LLM call was launched.
- Read-only plan and implementation reviews found no blocking issue. Ranking diagnostics in `rounds.json` are recorded for inspection; the current validator checks operator declarations, graph/action/prediction artifacts and pairing, and does not replay global scoring. The small real-loop fixture does not exercise a complete multi-round GB search; full experiments remain author-run.
