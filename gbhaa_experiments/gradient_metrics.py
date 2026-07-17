"""Dependency-light metrics for coarse-versus-fine gradient rows."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _finite_correlation(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def summarize_rows(rows: Sequence[Dict[str, object]], topk: int = 5) -> Dict[str, object]:
    if len(rows) < 2:
        raise ValueError("At least two feasible ball pairs are required")
    coarse = np.asarray([row["coarse_score"] for row in rows], dtype=float)
    fine = np.asarray([row["fine_max_score"] for row in rows], dtype=float)
    spearman = spearmanr(coarse, fine).statistic
    kendall = kendalltau(coarse, fine).statistic
    coarse_order = np.argsort(-coarse)
    fine_order = np.argsort(-fine)
    k = min(topk, len(rows))
    overlap = len(set(coarse_order[:k]).intersection(fine_order[:k])) / float(k)
    coarse_top = int(coarse_order[0])
    fine_top = int(fine_order[0])
    return {
        "pair_count": float(len(rows)),
        "spearman": _finite_correlation(spearman),
        "kendall": _finite_correlation(kendall),
        "action_agreement": float(
            np.mean([bool(row["action_agreement"]) for row in rows])
        ),
        "top1_hit": float(coarse_top == fine_top),
        "topk": float(k),
        "topk_overlap": float(overlap),
        "search_regret": float(fine[fine_top] - fine[coarse_top]),
        "selected_fine_score": float(fine[coarse_top]),
        "best_fine_score": float(fine[fine_top]),
    }
