# Projected editing factorial

Implement the user's approved two changes, without changing the thesis or running real experiments.

Four new opt-in profiles: edit_control (frequency words/random candidate), edit_gain (frequency words/projected gain), edit_words (gradient words/random candidate), edit_both (gradient words/projected gain). Each runs Post, Pre, Parallel with the same node-selection rule, structural prefix, budgets, training and split protocol. New reference is edit_control; old profiles remain byte-for-byte equivalent.

All four profiles issue exactly three generation requests per round, one template per request, 50 output tokens each, with word-plan caps 12/24/36. Four selected nodes are edited in ascending ID order. Each candidate passes text adaptation, vectorization, addition cap and the same cosine projection before scoring. All cells compute all candidate losses. Gain selection maximizes actual fixed-surrogate NLL on fixed clean pseudo-labels at non-training nodes; random selection uses seed/round/node-specific RNG. Negative gains are retained and reported; no hidden no-op or success filtering. Duplicate templates/candidate vectors are counted honestly. This new three-request control is not the historical one-request baseline.

Scoring and gradient graphs follow the arm: Post uses post-flip graph, Pre and Parallel use pre-flip graph. Sequential candidate losses include earlier accepted node writes. Fixed surrogate and pseudo-labels remain frozen. Validation outcomes do not enter candidate selection.

Gradient policy plans positive-gradient additions to absent mapped columns and negative-gradient removals from present mapped columns. Text adaptation must not reinsert planned deletions. Required words are verified through the actual vectorizer; unsupported mappings fail clearly. Projection may make removal fractional, so record actual changes. Common policy preserves old frequent-word logic. Existing partial-vocabulary alignment remains common to all cells.

Save raw requests/responses, word plans, candidate vectors/hashes, projections, losses, choices and realized round gains. Save the fixed GCN weights and vectorizer snapshot for independent offline replay of the editor audit. Audit graph routing, pseudo-label targets, vectorization, projection, sequential scoring and choices against persisted feature updates. Generation cache must match the saved request/response traces.

Quick launcher defaults to Citeseer seed 15, 60 actions, 12 trials, new result directory. Summary reports matched Post comparisons, per-arm factorial effects and interaction, sample SD and paired 95% CI; one seed has no interval and does not support stability claims. No fitting, network generation or full experiments during implementation tests.

## Clarifications after review
- Node selection uses the existing baseline AE-PR rule (alpha .28, 60 updates), applied to each cell's own current features and arm graph/restarts. It does not freeze/share Post's selected nodes across arms or across factorial cells.
- At round start, compute one feature gradient and freeze all node-specific word plans before any candidate generation or writes. Three shared templates serve all four nodes. Each slot's cap (12,24,36) independently bounds additions and removals per node; keep cap is 80. Exactly three requests per round, not per node. Choice factor cannot change that round's plans or requests given identical starting state.
- Factorial ACC contrasts, in percentage points: gain effect = ((control-gain)+(words-both))/2; word effect = ((control-words)+(gain-both))/2; interaction = gain+words-control-both. Positive favors the modification. Compute each within seed/arm first, then mean/sample SD/t CI; incomplete factorial blocks cannot supply full effects.
- Tests must cover all-negative candidate gains, duplicate candidates without resampling, nonzero sequential writes, request counts and loss-independent random choices.

Review clarification: caps bound planned additions/removals, not all final decreases. Legacy retention (80 words, continuous >0.1 threshold) and partial-vocabulary padding may cause other decreases; record unplanned increased/decreased entries separately and score the full final vector. Do not claim every final changed entry was gradient-directed.
