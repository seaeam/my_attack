# AE-PR Follow-up Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent audit/review and execute the coupled attack changes locally. No full attacks or network generation may be run by the assistant.

**Goal:** Supply reproducible experiment scripts, an evidence audit, matched selector controls and paired multi-seed summaries, leaving execution to the author.

**Architecture:** Preserve legacy result files. Per the author's explicit follow-up instruction, fix the default heir.py order to edge-first. Add a versioned follow-up protocol: degree-source global accumulation; edge-first endpoint diffusion; common all-node selector universe; deterministic node-ID ties and independent random streams. Recompute identical structural trajectories with frozen structure features and per-round structural seeds, then verify their hashes across selector arms; fail closed on mismatch. The full global/local factorial retains its own structural trajectories.

**Tech Stack:** Existing Python / NumPy / SciPy / PyTorch attack pipeline; stdlib CLI and JSON; unittest; shell launchers.

## Confirmed constraints

- The initial audit found attribute selection BEFORE the edge flip. The author explicitly requested fixing the default order first; this is now corrected and covered by a synthetic real-loop test. New results have a separate protocol/version and cannot be pooled with old results.
- Global legacy P @ z accumulation with uniform source is invariant; degree runs are not automatically invalid. Do not globally replace this operator with stationary PPR.
- Existing random/degree selectors use a different candidate pool. New four-arm selector comparisons share all nodes before identical budget filtering.
- Existing victim training consumes branch-dependent RNG state. New evaluations reset model RNG and save aligned predictions and node IDs.
- Old configurations and results are incomplete provenance, not automatic replacements for manuscript figures.

## Tasks

- [x] Audit old result.json files (not duplicate runs.jsonl): source/settings, missing seed-budget-variant cells, metric inconsistencies; write sanitized JSON/Markdown. Review plan independently.
- [x] Add failing synthetic tests for edge-first adjacency, matching pools, diffusion direction, deterministic ties, split RNG and paired statistics.
- [x] Add opt-in hooks in heir.py/meta.py and a new follow-up entrypoint: explicit configuration, round/action traces, model seeds, post-flip embeddings, generation seed/temperature, graph/split/feature hashes, exact metrics and completed budgets. Preserve unrelated defaults; edge-first is the explicitly requested default fix.
- [x] Add dry-run-by-default launcher for full new 2x2, strict selectors, pre-flip diagnostic, and Cora uniform-vs-degree repair profiles. Never overwrite or mix old results; require --execute and fingerprint-checked resume.
- [x] Add paired aggregation (complete blocks only, mean/sample SD, CIs, pairwise differences, 2x2 interaction, explicit missing cells); emit CSV/JSON/Markdown for review, not automatic paper replacement.
- [x] Run synthetic/unit tests and CLI dry-runs only. Independently review integration, fix defects, document exact user commands and evidence limits.

## File boundaries

- `gbhaa_experiments/followup_audit.py`: historical evidence inventory, standalone stdlib CLI.
- `gbhaa_experiments/followup_protocol.py`: deterministic selection, states, evaluation/artifact helpers.
- `gbhaa_experiments/entrypoints/meta_followup.py`: opt-in attack subclass and entrypoint.
- `gbhaa_experiments/run_followup.py`: versioned configuration matrix and runner.
- `gbhaa_experiments/summarize_followup.py`: paired statistics and comparability validation.
- `gbhaa_experiments/launchers/run_followup.sh`, `FOLLOWUP_EXPERIMENTS_ZH.md`: author-facing execution.
- `tests/test_followup_*.py`: bounded synthetic and schema tests.

## Validation commands

`/Users/shemingcong/anaconda3/envs/smc/bin/python -m unittest discover -s tests -p 'test_followup_*.py' -v`

`python -m gbhaa_experiments.run_followup --suite selectors` must plan only and perform no training/network calls.

`python -m gbhaa_experiments.followup_audit` must only read legacy data and write derived reports.

No commits or manuscript number substitutions are part of this task.

## Final verification

72 automatic tests passed (64 follow-up and 8 existing regression tests); four suite dry runs, entrypoint argument parsing, Python/shell syntax, whitespace checks and synthetic plot generation passed. No full attack, training run or real LLM request was executed. Legacy results and manuscript numbers were not changed.
