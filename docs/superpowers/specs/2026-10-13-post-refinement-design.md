# Post refinement design

## Authorized scope

Continue the user's existing opt-in Post tuning workflow. Prepare code and commands for the user to execute; do not launch training or generation, modify the thesis, replace old results, or promise a winning outcome. The user has already authorized tuning and requested 8–12 candidate configurations. Retain the shared structural trajectory and all three arm definitions.

## Evidence and alternatives

The 60-action Citeseer seed-15 sweep found alpha050 tied baseline ACC, winning only the NLL tie-break. Large floors, strong visit penalties and low restart probabilities did not improve ACC in that sweep. In baseline Post, 16/60 actions have zero cosine weight and 5/60 change the selected node set in same-feature pre/post probes.

Three possible next steps are: a focused local parameter search (selected here), larger generation budgets (confounds attack workload), or a new selector that explicitly scores graph changes (a new method requiring separate controls). This iteration reuses the existing single-graph scorer and generation/projection budget.

## Search design

Add a `refine` preset of ten profiles: baseline, alpha050, alpha040, alpha060, a50_f01, a50_f03, a50_f05, a50_diffused, a50_f03_diffused, a50_coverage025. New profiles all have 60 diffusion iterations. `a50_f01/f03/f05` use alpha=.5 and weight floors .01/.03/.05; diffused removes direct restart mass; coverage025 uses alpha=.5 and gamma=.25. All generation, projection, node-write, structural and victim budgets stay equal. Existing quick/tiny/extended profile matrices and old configs stay unchanged.

Every new profile is applied to all three arms. Screening defaults to Citeseer seed 15, first 60 actions: 30 short trials. Rank candidates by Post validation ACC ascending, then NLL descending, as before. Do not select by making controls worse or by choosing the largest observed margin.

Confirmation freezes the selected config, runs seeds 16–19 and 120-action prefixes, and includes all three matched arms plus baseline Post (16 trials, or 12 if baseline wins). These are additional development-seed checks, not a pristine test set. Do not pool 60- and 120-action results. A missing/invalid result suppresses recommendations.

## Analysis and commands

Extend paired comparison reporting with ACC-only win/loss/tie counts and seed-level paired Student-t 95% intervals. Preserve existing ACC-then-NLL counts for compatibility, with explicit labeling. Single-seed intervals are undefined. Mark exploratory asymmetric comparisons and incomplete matrices clearly; do not automatically label any profile stable from development selection. Display paired comparisons in Markdown and export CSV.

Add a launcher with screen (default), confirm, summary-screen and summary-confirm modes. Preserve environment resolution and forwarding of run options through the existing launcher. New output/report roots isolate this refinement from completed tuning. Default is preview; only --execute runs experiments. Confirmation consumes the saved recommendation and rejects screening-seed reuse using existing runner checks.

## Verification

Test the preset size, legacy compatibility, identical configs across matched arms, unchanged generation budgets, CI and ACC tie semantics, wrapper argument routing and no-execution defaults. Exercise new profile replay on synthetic graphs, run existing tuning/action-state regressions, preview real launch commands, and revalidate old completed reports without overwriting them. Optional saved-state scoring diagnostics use no labels, training or LLM calls and must not be represented as attacked-ACC predictions.
