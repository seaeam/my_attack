# Standard PR Selector Implementation Plan

**Goal:** Implement the user-approved AE-PR / ordinary PR / Degree / Random comparison on the existing pre-flip graph, preserving endpoint PPR as a separate factorial ablation.

**Architecture:** Add a distinct `pr` strategy using NetworkX PageRank with uniform restart/dangling redistribution and unweighted topology. Match the restart probability to local AE-PR (`0.28`, NetworkX damping `0.72`); use explicit convergence limits. Version only the selectors suite and route its default output to a new directory. Keep other suite definitions, old artifacts and the main attack order unchanged.

**Tech Stack:** Existing Python, NetworkX, SciPy, NumPy, PyTorch, unittest and shell launchers. No training, full attacks, network generation, commits or manuscript edits are part of this implementation.

## Tasks

- [x] Add failing tests in `tests/test_followup_pr.py`: analytic nonuniform rank, isolated nodes, endpoint/feature independence, distinction from endpoint PPR, convergence failure, pre-flip selector jobs, preserved factorial definitions and isolated output/protocol.
- [x] Implement `pr` in `gbhaa_experiments/followup_protocol.py`; expose it in `entrypoints/meta_followup.py`, record the solver settings, and write the requested protocol to artifacts.
- [x] Update `run_followup.py` to use PR and pre only for selectors, with a suite-specific protocol and default output; retain legacy defaults for other suites.
- [x] Update `summarize_followup.py`, summary fixtures and plotting labels so the new protocol validates through to a checked report; reject legacy/post selector records.
- [x] Update author instructions with exact dry-run, execution, summary and plotting commands and interpretation limits.
- [x] Run analytic/synthetic tests, positive and rejection summary fixtures, legacy regressions, dry runs and syntax/diff checks. Four pre-existing edge-order tests are deferred at the user's request and must be reported explicitly if still failing.

## Validation

Use `/Users/shemingcong/anaconda3/envs/smc/bin/python -m unittest discover -s tests -p 'test_followup_*.py' -v` and focused attribute-budget/streaming regression tests. Use a temporary output directory for synthetic reports. Do not rewrite existing experiment results or launch full runs.

## Results

- Follow-up suite: 78 tests, 74 passed. The same four pre-existing edge-first tests fail (three missing `prepare_attribute_adjacency`, one expecting a flipped graph during selection). The user explicitly deferred changing that loop order.
- Attribute-node-budget and streaming regressions: 10 passed.
- A four-arm synthetic run verified the real loop, selector, action trace, protocol metadata, evaluation writer and resume checks without training or generation. A two-seed artifact fixture passed summary and SVG/PNG rendering with the new PR contrast and label.
- Launcher dry-run, Python compilation and diff whitespace checks on changed implementation/test/documentation files passed. The repository-wide whitespace check flags a pre-existing trailing blank line in a user experiment log; that log was left intact.
- Read-only implementation review found no blocking defects. New report paths are explicit in the author commands. No full experiment, LLM request, manuscript change or result migration was performed.
