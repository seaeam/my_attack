# Post transfer diagnostic and matched pilot

## Evidence and purpose

Citeseer seed 15, 60 actions: edit_both Post damages its fixed linear GCN
(validation ACC 60.6635%) but retrained ReLU victim ACC is 69.1943%, worse
than edit_control 66.8246%. Of its 6,708 unplanned decreases, 6,598 occur
outside the 500-word vocabulary (feature width 3,703). These are diagnostic
observations, not proof of a remedy. Gain/L1 would change 23/240 choices;
defer that and unrolled retraining until this smaller experiment resolves the
more direct model and edit-support discrepancies.

## Design

Add three opt-in profiles, all with gradient words and projected-NLL selection:
`transfer_relu`, `transfer_residual`, `transfer_both`. Compare them with unchanged
`edit_both` (factorial anchor) and `edit_control` (previous best Post reference).
Default 5 profiles x 3 arms, Citeseer seed 15, 60 actions, in a new result root.

Factor A trains a separate clean CPU ReLU GCN, with the victim's architecture
and no-validation 200-epoch fitting recipe, but an independent deterministic
`editor-transfer` seed. It never replaces the structural surrogate. Supply only
training node IDs and training labels to the fitter. Restore ambient RNGs.
Keep original fixed pseudo-labels and all-nontraining targets across profiles.
This factor changes the architecture/fitting/initialization bundle, not ReLU
alone. Use the new model for both word gradients and sequential projected candidate
NLL on the arm graph. Snapshot both source and new models, clean predictions,
training inputs/recipe hashes; audit their clean forwards without refitting.
This tests clean proxy transfer; it does not simulate victim retraining.

Factor B starts from the current feature row and applies only the planned
additions/removals supported by the actual vectorizer. Preserve all unplanned
coordinates, including the unmapped tail and fractional active values. Apply
the existing added-word/cosine projection, then restore unchanged coordinates
exactly to avoid floating-point drift. This is a partial-vocabulary feature
patch, not a full text replacement and not a repair of unknown column semantics.
Save policy, scope counts, actual changes/L1 and usual candidate audit data.

All arms receive the same policy. Post uses executed post graph; Pre uses the
pre graph with the same endpoints; Parallel uses the pre graph with independent
restarts. Graph for gradients/scoring follows that same arm. Node selection,
generation (3 requests, 3 candidates, caps 12/24/36), four writes, visits,
projection bounds, labels/targets, structural trajectory and victim recipe
remain matched. Actual node selections may diverge as edited features affect
future diffusion. Actual edit amplitudes need not match and must be reported.

## Integration / safeguards

- All 37 existing profile dictionaries/operators and historical audits remain
  valid. Never change heir.py, meta.py, prior replay ordering or thesis.
- Separate preset `transfer`; shared editor reference remains edit_control.
- Model identity must match across all arms and profiles with the same model
  policy. Vocabulary/source pseudo-label identity must match the whole block.
- Exact replay of candidates, graph/selection, requests, projection and chosen
  writes remains mandatory. Fail closed on inconsistent checkpoint/recipe.
- Add a separate transfer 2x2 table (edit_both, relu, residual, both). Compute
  differences per seed/arm, then means, sample SD and t CI. Incomplete planned
  matrices suppress estimates. n=1 has no SD/CI. Reusing development seeds
  does not establish held-out or stable superiority.
- Launcher preview performs no experiment and creates no result root. Explicit
  execute is left to the user. Optional paired seeds 15/16 require a new root.

## Verification

Test partial-vocabulary invariance, exact unplanned-coordinate preservation
after projection, gradient plan execution, source model/RNG preservation,
train-only fitting, all profile/arm dispatches, checkpoint/trace tampering,
factorial missing cells, matched identities, public report path and launcher.
Run relevant offline suite plus representative old artifact audits. No real
LLM calls or dataset training during implementation; tiny synthetic fits only.
