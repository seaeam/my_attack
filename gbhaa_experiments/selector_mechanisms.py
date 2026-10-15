"""Single-graph selector variants. No fitting, ground-truth labels or generation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from gbhaa_experiments.post_tuning_protocol import (
    diffusion_scores, validate_config, _visit_counts, COVERAGE_SELECTOR_RULES,
    GUIDED_SELECTOR_RULES, selector_version,
)

VERSION = "endpoint-frontier-v1"
RULES = ("frontier", "balanced_frontier", *GUIDED_SELECTOR_RULES)


def compute_states(features, adj, restarts, config):
    config = validate_config(config)
    restarts = list(restarts)
    sources = [[u] for u in restarts] if config["selector_rule"] == "balanced_frontier" else [restarts]
    return np.stack([diffusion_scores(features, adj, seeds, config) for seeds in sources])


def order_from_states(states, restarts, visits, rule, pool_size, gains=None):
    """Full permutation, prioritizing legal frontier nodes and explicit fallback.

    Balanced rows follow the supplied restart order. Sorting sources internally
    makes swapping the undirected endpoints leave the final ranking unchanged.
    """
    states = np.asarray(states, dtype=np.float32)
    if (rule not in RULES or states.ndim != 2 or states.shape[0] != (2 if rule == "balanced_frontier" else 1)
            or states.shape[1] < 4 or not np.isfinite(states).all() or np.any(states < 0)):
        raise ValueError("Invalid mechanism rule or diffusion states")
    n = states.shape[1]
    seeds = np.asarray(restarts)
    if seeds.shape != (2,) or seeds.dtype.kind not in "iu" or len(set(seeds.tolist())) != 2 or np.any((seeds < 0) | (seeds >= n)):
        raise ValueError("Mechanisms require two distinct valid restart IDs")
    if type(pool_size) is not int or not 4 <= pool_size <= 256:
        raise ValueError("Invalid candidate pool size")
    counts = _visit_counts(visits, n)
    legal = counts < 10
    if legal.sum() < 4:
        raise ValueError("Fewer than four legal nodes remain")
    scores = states.mean(axis=0)
    ids = np.arange(n)
    full = np.lexsort((ids, -scores)).tolist()
    nonseed = legal.copy(); nonseed[seeds] = False
    frontier = [u for u in full if nonseed[u] and scores[u] > 0]
    pool = []; zero_gain = False
    if rule == "balanced_frontier":
        branches = []
        for row in states[np.argsort(seeds)]:
            branches.append([int(u) for u in np.lexsort((ids, -row)) if nonseed[u] and row[u] > 0])
        preferred = []; seen = set(); positions = [0, 0]
        while any(positions[k] < len(branches[k]) for k in (0, 1)):
            for k in (0, 1):
                while positions[k] < len(branches[k]) and branches[k][positions[k]] in seen:
                    positions[k] += 1
                if positions[k] < len(branches[k]):
                    u = branches[k][positions[k]]; positions[k] += 1
                    seen.add(u); preferred.append(u)
    elif rule in GUIDED_SELECTOR_RULES:
        gains = np.asarray(gains)
        if gains.shape != (n,) or not np.isfinite(gains).all() or np.any(gains < 0):
            raise ValueError("Edit gains must be finite, nonnegative and match the node count")
        pool = frontier[:pool_size]
        zero_gain = not pool or not np.any(gains[pool] > 0)
        adjusted_gains = gains / (1 + counts) if rule == "frontier_editgain_decay" else gains
        ranked = sorted(pool, key=lambda u: (-float(adjusted_gains[u]), -float(scores[u]), u))
        if rule == "frontier_editgain_fresh3":
            # A minimum reservation within the original local pool. The fourth
            # node may also be fresh. Zero gains still retain this reservation.
            reserved = [u for u in ranked if counts[u] == 0][:3]
            reserved_set = set(reserved)
            ranked = reserved + [u for u in ranked if u not in reserved_set]
        preferred = ranked + frontier[len(pool):]
    else:
        preferred = frontier
    seen = set(preferred)
    order = preferred + [u for u in full if legal[u] and u not in seen] + [u for u in full if not legal[u]]
    selected = order[:4]
    info = dict(candidate_pool=pool, positive_nonrestart_count=len(frontier), selected_nodes=selected,
                fallback_count=sum(u not in set(frontier) for u in selected), zero_gain_fallback=zero_gain)
    if rule in GUIDED_SELECTOR_RULES:
        info["selected_gains"] = [float(gains[u]) for u in selected]
    if rule in COVERAGE_SELECTOR_RULES:
        fresh_pool = sum(counts[u] == 0 for u in pool)
        info.update(fresh_pool_count=int(fresh_pool), selected_fresh_count=int(sum(counts[u] == 0 for u in selected)),
                    selected_prior_visits=[int(counts[u]) for u in selected],
                    selected_adjusted_gains=[float(adjusted_gains[u]) for u in selected])
        if rule == "frontier_editgain_fresh3":
            info.update(fresh_quota_requested=3, fresh_quota_effective=min(3, int(fresh_pool)),
                        fresh_quota_shortfall=max(0, 3-int(fresh_pool)))
    return np.asarray(order, dtype=np.int64), scores, info


def edit_potential(gradient, features, vocabulary_width, max_added_words):
    """First-order opportunity proxy; actual text edits need not realize it.

    Remove beneficial current coordinates; add unit words only in absent,
    vocabulary-supported coordinates, respecting the generator's addition cap.
    """
    gradient, features = np.asarray(gradient), np.asarray(features)
    if (gradient.ndim != 2 or gradient.shape != features.shape or not np.isfinite(gradient).all()
            or not np.isfinite(features).all() or np.any(features < 0)
            or type(vocabulary_width) is not int or not 0 < vocabulary_width <= features.shape[1]
            or type(max_added_words) is not int or max_added_words < 1):
        raise ValueError("Invalid gradient/features or edit limits")
    removal = np.maximum(-gradient * features, 0).sum(axis=1, dtype=np.float64)
    additions = np.where(features[:, :vocabulary_width] == 0, np.maximum(gradient[:, :vocabulary_width], 0), 0)
    k = min(max_added_words, vocabulary_width)
    return removal + np.partition(additions, vocabulary_width-k, axis=1)[:, -k:].sum(axis=1, dtype=np.float64)


def feature_loss_gradient(model, features, adj_norm, pseudo_labels, targets):
    """Differentiate fixed pseudo-label NLL w.r.t. X without changing model state."""
    x = features.detach().clone().requires_grad_(True)
    pseudo = torch.as_tensor(pseudo_labels, dtype=torch.long, device=x.device)
    ids = torch.as_tensor(targets, dtype=torch.long, device=x.device)
    if pseudo.shape != (len(x),) or ids.ndim != 1 or not len(ids) or torch.any(ids < 0) or torch.any(ids >= len(x)):
        raise ValueError("Invalid fixed pseudo-labels or gradient targets")
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.enable_grad():
            output = model(x, adj_norm)
            loss = F.nll_loss(output[ids], pseudo[ids])
            gradient, = torch.autograd.grad(loss, x)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite selector surrogate loss or gradient")
        return gradient.detach().cpu().numpy(), float(loss.detach())
    finally:
        for module, mode in modes:
            module.training = mode
