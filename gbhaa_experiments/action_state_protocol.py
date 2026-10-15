"""Fixed rules for the isolated matched structural-action experiment."""
from __future__ import annotations

import hashlib
import json
import math

PROTOCOL = "gbca-action-state-v1"
ARMS = ("parallel", "endpoint_pre", "endpoint_post")
DATASETS = ("citeseer", "cora", "pubmed")


def protocol_settings():
    from gbhaa_experiments.run_followup import settings
    values = settings({"suite": "selectors", "global_source": "degree", "search_method": "gb"})
    values["text_attack_nodes"] = 4
    return values


def operator_settings():
    return {
        "global_source": "unweighted_degree_on_current_graph",
        "global_restart": .15, "global_accumulated_states": 30,
        "local_restart": .28, "local_updates": 60,
        "local_update": "alpha*q + (1-alpha)*(P.T@state)",
        "local_readout": "last_state", "attribute_weights": "clipped_cosine",
        "zero_weight_rows": "self_loop", "ties": "node_id_ascending",
        "parallel_restart": "independent_uniform_pair_per_round_allow_endpoint_overlap",
        "structure_features": "frozen_original", "structure_actions": "saved_replay",
        "grouping_graph": "post_in_all_arms", "diffusion_features": "current_arm_features",
        "nodes_per_round": 4, "max_visits_per_node": 10, "distinct_node_cap": None,
        "generation_cache": "full_request_and_settings", "template_cache": "reset_each_round",
        "generation_max_tokens": 50,
    }


def make_jobs(datasets, seeds, budgets):
    datasets, seeds, budgets = list(datasets), list(seeds), list(budgets)
    if (not datasets or not seeds or not budgets or len(set(datasets)) != len(datasets)
            or len(set(seeds)) != len(seeds) or len({f"{b:g}" for b in budgets}) != len(budgets)
            or any(d not in DATASETS for d in datasets)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or any(not math.isfinite(b) or not 0 < b <= 1 for b in budgets)):
        raise ValueError("Use supported, distinct datasets, distinct nonnegative seeds and budgets in (0,1]")
    return [{"dataset": d, "seed": s, "budget": float(b), "variant": v}
            for d in datasets for b in budgets for s in seeds for v in ("structure", *ARMS)]


def independent_restarts(n_nodes, seed, round_index):
    import numpy as np
    from gbhaa_experiments.followup_protocol import seed_stream
    if n_nodes < 2:
        raise ValueError("Independent pair requires at least two nodes")
    rng = np.random.default_rng(seed_stream(seed, "independent_restart", round_index))
    return sorted(int(v) for v in rng.choice(n_nodes, 2, replace=False))


def actions_hash(actions):
    return hashlib.sha256(json.dumps(actions).encode()).hexdigest()


def flip_action(adj, action):
    if (not isinstance(action, list) or len(action) != 3
            or any(type(v) is not int for v in action)):
        raise ValueError("Actions must be integer [u,v,new_value] triples")
    u, v, value = action
    if (u == v or not 0 <= u < adj.shape[0] or not 0 <= v < adj.shape[0]
            or value not in (0, 1) or adj[u, v] not in (0, 1)
            or adj[u, v] != adj[v, u] or value != 1 - adj[u, v]):
        raise ValueError("Action must flip one binary, symmetric, non-self edge")
    post = adj.tolil(copy=True)
    post[u, v] = post[v, u] = value
    post = post.tocsr(); post.eliminate_zeros()
    return post


def validate_trajectory(original, modified, actions, target):
    import numpy as np
    if (original.shape != modified.shape or original.shape[0] != original.shape[1]
            or (original != original.T).nnz or np.any(original.diagonal())
            or not np.isfinite(original.data).all() or not np.isin(original.data, [0, 1]).all()
            or type(target) is not int or target < 1 or len(actions) != target):
        raise ValueError("Invalid trajectory graph or edge budget")
    state = original.copy(); seen = set()
    for action in actions:
        state = flip_action(state, action)
        edge = tuple(sorted(action[:2]))
        if edge in seen:
            raise ValueError("Trajectory reflips an already edited edge")
        seen.add(edge)
    if (state != modified).nnz or (original != modified).nnz != 2 * target:
        raise ValueError("Trajectory replay differs from saved final graph")


def rank_on_graph(attack, features, adj, restarts):
    from gbhaa_experiments.followup_protocol import stable_order
    _, scores = attack.ppr_topk_from_seeds(
        fea=features, adj=adj, seed_nodes=restarts, topk=adj.shape[0],
        alpha=.28, N=60, use_cos_sim=True,
    )
    return stable_order(scores), scores


def feature_measures(original, modified):
    import numpy as np
    from gbhaa_experiments.followup_protocol import as_array
    original, modified = as_array(original), as_array(modified)
    if original.shape != modified.shape or original.ndim != 2 or not np.isfinite(modified).all():
        raise ValueError("Invalid modified feature array")
    delta = modified.astype(np.float64) - original.astype(np.float64)
    changed = delta != 0
    return {"changed_feature_nodes": int(changed.any(axis=1).sum()),
            "changed_feature_entries": int(changed.sum()),
            "feature_l1": float(np.abs(delta).sum()),
            "feature_l2": float(np.sqrt(np.square(delta).sum())),
            "feature_linf": float(np.abs(delta).max(initial=0))}
