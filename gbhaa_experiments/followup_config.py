"""Protocol identifiers and fixed solver settings (safe for dry-run imports)."""

PROTOCOL = "aepr-followup-v1-edge-first"
SELECTOR_PROTOCOL = "aepr-selectors-v2-pre-pr"
FACTORIAL_PROTOCOL = "aepr-factorial-v2-pre-stable"
PR_TOL = 1e-12
PR_MAX_ITER = 1000


def protocol_for_suite(suite):
    return {"selectors": SELECTOR_PROTOCOL, "factorial": FACTORIAL_PROTOCOL}.get(suite, PROTOCOL)


def factorial_operator_settings(global_alpha=.15, global_states=30,
                                local_alpha=.28, local_updates=60):
    """The fixed operator rules; weighting switches are recorded separately."""
    return {
        "global": {
            "source": "unweighted_degree_on_current_graph",
            "initialization": "alpha * source",
            "update": "(1-alpha) * (P @ state) + alpha * source",
            "restart_probability": global_alpha,
            "accumulated_states": global_states,
            "readout": "sum_states_including_initial_then_l1_normalize",
            "features": "frozen_original",
        },
        "local": {
            "source": "equal_mass_on_current_edge_endpoints",
            "initialization": "source",
            "update": "alpha * source + (1-alpha) * (P.T @ state)",
            "restart_probability": local_alpha,
            "updates": local_updates,
            "readout": "last_state",
            "features": "current_modified_features",
        },
        "transition_normalization": "row",
        "attribute_weights": "cosine_clipped_to_0_1",
        "zero_weight_rows": "self_loop",
        "global_ties": "node_id_ascending",
        "local_ties": "node_id_ascending",
        "local_graph": "pre",
        "structure_features": "frozen_original",
    }


def pr_solver_settings(restart_probability):
    return {
        "algorithm": "networkx.pagerank",
        "restart_probability": restart_probability,
        "damping": 1.0 - restart_probability,
        "personalization": "uniform",
        "dangling": "uniform",
        "weight": None,
        "tol": PR_TOL,
        "max_iter": PR_MAX_ITER,
    }
