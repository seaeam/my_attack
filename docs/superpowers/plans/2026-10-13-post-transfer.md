# Post transfer pilot implementation plan

**Goal:** Deliver a reproducible 15-trial matched pilot for model alignment and
planned-feature preservation, with no thesis changes or actual experiment run.

**Architecture:** Extend existing opt-in editor with two explicit factors;
separate training/snapshot audit code in `editor_transfer.py`. Preserve all old
profile semantics. Reuse public runner and strict artifact replay audit.

**Tech stack:** Python, NumPy/SciPy, PyTorch/DeepRobust GCN, unittest, Bash.

- [x] Add failing tests for partial-vocabulary preservation/projection, model
  training isolation, profile/preset matching and launcher preview.
- [x] Add `editor_transfer.py` for train-only seeded clean ReLU proxy and saved
  checkpoint/clean-output provenance audit; integrate prepare_editor and entry.
- [x] Add residual adaptation flag to projected_editing.py and runtime dispatch;
  preserve legacy default outputs/traces and add new scope diagnostics.
- [x] Add 3 catalog profiles and transfer preset; strict schema/operator fields;
  group editor snapshot comparisons by the intended model factor only.
- [x] Extend editor audits, transfer factorial summaries, per-trial diagnostics,
  launcher and Chinese running guide. Keep old editor factorial unchanged.
- [x] Test every new arm and both factors, public summary, tamper rejection,
  absent cells, seed pairing and no-I/O preview; run existing offline tests.
- [x] Review implementation; compare old catalog/operators/protected hashes and
  audit representative saved runs. Write QA and diagnostic artifacts.

Tests use `/Users/shemingcong/anaconda3/envs/smc/bin/python -m unittest discover
-s tests -p '<pattern>'`. No commits, worktree switches or real experiments.
The user has explicitly selected diagnosis followed by implementation; routine
design choices stay within this authorization. Request an independent spec/
plan review as required by the applied planning skills, then implement inline.

Verification: 160 relevant offline tests passed; 37 old profiles/operators unchanged, 118 historical job identities valid, six representative historical artifact audits passed. Both independent-review findings were fixed and their two regression tests passed independent recheck. Real experiments were not run.
