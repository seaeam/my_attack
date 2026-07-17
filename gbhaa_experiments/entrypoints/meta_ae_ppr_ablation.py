"""Clean 2x2 global/local PPR versus AE-PPR comparison."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import heir as heir_module


REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_ATTACK = heir_module.Heirattack
_GLOBAL_TRANSITION = "ae_ppr"
_LOCAL_TRANSITION = "ae_ppr"


class AEPPRComparisonAttack(_BASE_ATTACK):
    def __init__(self, *args, **kwargs):
        patched_class = heir_module.Heirattack
        heir_module.Heirattack = _BASE_ATTACK
        try:
            super().__init__(*args, **kwargs)
        finally:
            heir_module.Heirattack = patched_class

    def compute_global_ppr(self, *args, **kwargs):
        kwargs["use_cos_sim"] = _GLOBAL_TRANSITION == "ae_ppr"
        return super().compute_global_ppr(*args, **kwargs)

    def select_text_candidate_nodes(self, fea, adj, row_idx, col_idx):
        k = self._text_candidate_budget(adj.shape[0])
        selected, _ = self.ppr_topk_from_seeds(
            fea=fea,
            adj=adj,
            seed_nodes=[row_idx, col_idx],
            topk=k,
            alpha=self.text_ppr_alpha,
            N=self.text_ppr_iters,
            use_cos_sim=_LOCAL_TRANSITION == "ae_ppr",
        )
        return selected


def _consume_args() -> None:
    global _GLOBAL_TRANSITION, _LOCAL_TRANSITION
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--experiment_global_transition",
        choices=["ppr", "ae_ppr"],
        required=True,
    )
    parser.add_argument(
        "--experiment_local_transition",
        choices=["ppr", "ae_ppr"],
        required=True,
    )
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    _GLOBAL_TRANSITION = parsed.experiment_global_transition
    _LOCAL_TRANSITION = parsed.experiment_local_transition
    sys.argv = [sys.argv[0], *remaining, "--local_candidate_strategy", "local_ae_ppr"]


def main() -> None:
    _consume_args()
    print(
        "AE-PPR comparison: "
        f"global={_GLOBAL_TRANSITION}, local={_LOCAL_TRANSITION}"
    )
    heir_module.Heirattack = AEPPRComparisonAttack
    runpy.run_path(str(REPO_ROOT / "meta.py"), run_name="__main__")


if __name__ == "__main__":
    main()
