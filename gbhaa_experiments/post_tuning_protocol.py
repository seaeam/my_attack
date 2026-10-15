"""Profiles and single-graph scoring for the isolated Post development sweep.

The scorer only receives the graph selected by its arm. It never receives a
second graph, labels, or validation outcomes. Catalogs and jobs return fresh
JSON-compatible configurations, so a trial cannot modify another trial's setup.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real
import re

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from gbhaa_experiments.action_state_protocol import ARMS, DATASETS
from gbhaa_experiments.followup_protocol import stable_order

PROTOCOL = "gbca-post-tuning-v1"

_BASELINE = {
    "id": "baseline", "family": "baseline",
    "description": "Original endpoint diffusion and text-generation settings.",
    "alpha": .28, "iterations": 60, "weight_floor": 0., "visit_gamma": 0.,
    "readout": "last", "generation_max_tokens": 50,
    "text_min_cluster_size": 2, "text_max_cluster_size": 4,
    "text_max_added_words": 70, "text_cdl_topk": 36,
    "text_budget_per_node": 80, "text_similarity_min": .35,
}
_LEGACY_VARIANTS = {
    "baseline": ("baseline", _BASELINE["description"], {}),
    "alpha010": ("restart", "Restart probability .10 with 120 updates.", {"alpha": .10, "iterations": 120}),
    "alpha015": ("restart", "Restart probability .15 with 120 updates.", {"alpha": .15, "iterations": 120}),
    "alpha050": ("restart", "Restart probability .50.", {"alpha": .50}),
    "floor010": ("edge_floor", "A .10 floor on every existing edge weight.", {"weight_floor": .10}),
    "floor025": ("edge_floor", "A .25 floor on every existing edge weight.", {"weight_floor": .25}),
    "coverage05": ("coverage", "Divide readout by the square root of one plus visits.", {"visit_gamma": .5}),
    "coverage10": ("coverage", "Divide readout by one plus visits.", {"visit_gamma": 1.}),
    "diffused": ("readout", "Remove direct restart mass from the last state.", {"readout": "diffused"}),
    "combined": ("combined", "Restart .15, floor .10, visit exponent .5, propagated readout.",
                 {"alpha": .15, "iterations": 120, "weight_floor": .10,
                  "visit_gamma": .5, "readout": "diffused"}),
    "alpha005": ("restart", "Restart probability .05 with 240 updates.", {"alpha": .05, "iterations": 240}),
    "alpha075": ("restart", "Restart probability .75.", {"alpha": .75}),
    "floor050": ("edge_floor", "A .50 floor on every existing edge weight.", {"weight_floor": .50}),
    "coverage025": ("coverage", "Visit penalty exponent .25.", {"visit_gamma": .25}),
    "combo_coverage": ("combined", "Restart .15 with visit penalty exponent 1.",
                       {"alpha": .15, "iterations": 120, "visit_gamma": 1.}),
    "combo_floor": ("combined", "Edge floor .10 with visit penalty exponent 1.",
                    {"weight_floor": .10, "visit_gamma": 1.}),
    "tokens150": ("generation", "Generation variant with a 150-token response limit.", {"generation_max_tokens": 150}),
    "groups2": ("generation", "Generation variant using two-node text groups.", {"text_max_cluster_size": 2}),
    "added35": ("projection", "Projection variant limiting added words to 35.", {"text_max_added_words": 35}),
    "cdl18": ("projection", "Projection variant with CDL top-k 18.", {"text_cdl_topk": 18}),
}
_REFINEMENTS = {
    "alpha040": ("restart", "Restart probability .40.", {"alpha": .40}),
    "alpha060": ("restart", "Restart probability .60.", {"alpha": .60}),
    "a50_f01": ("refinement", "Restart .50 with edge floor .01.", {"alpha": .50, "weight_floor": .01}),
    "a50_f03": ("refinement", "Restart .50 with edge floor .03.", {"alpha": .50, "weight_floor": .03}),
    "a50_f05": ("refinement", "Restart .50 with edge floor .05.", {"alpha": .50, "weight_floor": .05}),
    "a50_diffused": ("refinement", "Restart .50 with propagated readout.", {"alpha": .50, "readout": "diffused"}),
    "a50_f03_diffused": ("refinement", "Restart .50, edge floor .03, propagated readout.",
                         {"alpha": .50, "weight_floor": .03, "readout": "diffused"}),
    "a50_coverage025": ("refinement", "Restart .50 with mild visit penalty exponent .25.",
                        {"alpha": .50, "visit_gamma": .25}),
}
_MECHANISMS = {
    name: ("selector_mechanism", description, {"selector_rule": name, "selector_pool_size": 32})
    for name, description in (
        ("frontier", "Prioritize positive-mass non-restart nodes; original ranking fallback."),
        ("balanced_frontier", "Interleave the two single-source non-restart rankings."),
        ("frontier_editgain", "Rerank up to 32 frontier nodes by clean-surrogate feature-edit potential."),
    )
}
_COVERAGE_MECHANISMS = {
    name: ("selector_mechanism", description, {"selector_rule": name, "selector_pool_size": 32})
    for name, description in (
        ("frontier_editgain_decay", "Same 32-node gain pool; rank by gain/(1+prior visits)."),
        ("frontier_editgain_fresh3", "Same gain pool; reserve up to three never-visited nodes before gain-ranked fill."),
    )
}
COVERAGE_SELECTOR_RULES = tuple(_COVERAGE_MECHANISMS)
GUIDED_SELECTOR_RULES = ("frontier_editgain", *COVERAGE_SELECTOR_RULES)
EDIT_PROFILES = ("edit_control", "edit_gain", "edit_words", "edit_both")
_EDIT_VARIANTS = {name: ("projected_editing", description, dict(editor_gain=gain, editor_gradient=gradient, editor_candidates=3))
    for name, gain, gradient, description in (
        ("edit_control", False, False, "Three real candidates; frequency words; seeded random choice."),
        ("edit_gain", True, False, "Frequency words; choose by actual projected surrogate loss."),
        ("edit_words", False, True, "Gradient-directed word edits; seeded random choice."),
        ("edit_both", True, True, "Gradient-directed word edits and projected-loss choice."))}
TRANSFER_PROFILES = ('edit_both', 'transfer_relu', 'transfer_residual', 'transfer_both')
_TRANSFER_VARIANTS = {
    name: ('editor_transfer', description, dict(editor_gain=True, editor_gradient=True,
        editor_candidates=3, editor_proxy=proxy, editor_residual=residual))
    for name,proxy,residual,description in (
        ('transfer_relu','relu',False,'Victim-aligned clean ReLU proxy; full replacement editing.'),
        ('transfer_residual','source',True,'Original proxy; preserve all unplanned feature coordinates.'),
        ('transfer_both','relu',True,'Victim-aligned proxy and planned feature patches.'))}
OBJECTIVE_PROFILES = ('transfer_relu', 'objective_margin', 'objective_local', 'objective_both')
_OBJECTIVE_VARIANTS = {
    name: ('editor_objective', description, dict(editor_gain=True, editor_gradient=True,
        editor_candidates=3, editor_proxy='relu', editor_residual=False,
        editor_objective=kind, editor_local_mix=mix))
    for name,kind,mix,description in (
        ('objective_margin','capped_margin',0.,'ReLU proxy; capped pseudo-class margin objective.'),
        ('objective_local','nll',.5,'ReLU proxy; half uniform, half arm-local diffusion target weights.'),
        ('objective_both','capped_margin',.5,'ReLU proxy; capped margin and arm-local target weights.'))}
_VARIANTS = {**_LEGACY_VARIANTS, **_REFINEMENTS, **_MECHANISMS, **_COVERAGE_MECHANISMS, **_EDIT_VARIANTS, **_TRANSFER_VARIANTS, **_OBJECTIVE_VARIANTS}
_PRESETS = {
    "quick": tuple(_LEGACY_VARIANTS)[:10],
    "tiny": ("baseline", "floor025", "coverage10", "combined"),
    "extended": tuple(_LEGACY_VARIANTS),
    "refine": ("baseline", "alpha050", *_REFINEMENTS),
    "mechanism": ("baseline", *_MECHANISMS),
    "coverage": ("baseline", "frontier_editgain", *_COVERAGE_MECHANISMS),
    "editing": EDIT_PROFILES,
    "transfer": ('edit_control', *TRANSFER_PROFILES),
    "objective": OBJECTIVE_PROFILES,
}


def selector_version(rule):
    return "endpoint-frontier-coverage-v1" if rule in COVERAGE_SELECTOR_RULES else "endpoint-frontier-v1"


def profile_catalog():
    """Return all complete, independent profile dictionaries in catalog order."""
    return {name: {**_BASELINE, "id": name, "family": family, "description": description, **changes}
            for name, (family, description, changes) in _VARIANTS.items()}


def get_profile(profile_id):
    """Return a fresh catalog profile; unrecognized IDs are an error."""
    if not isinstance(profile_id, str) or profile_id not in _VARIANTS:
        raise ValueError(f"Unknown post-tuning profile: {profile_id!r}")
    return profile_catalog()[profile_id]


def _sequence(values, name):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a nonempty sequence")
    try:
        result = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty sequence") from exc
    if not result:
        raise ValueError(f"{name} must be a nonempty sequence")
    return result


def select_profiles(preset="quick", profile_ids=None):
    """Select unique profiles; explicit IDs replace the preset's profile list."""
    if not isinstance(preset, str) or preset not in _PRESETS:
        raise ValueError(f"Unknown post-tuning preset: {preset!r}")
    selected = list(_PRESETS[preset]) if profile_ids is None else _sequence(profile_ids, "profile_ids")
    if any(not isinstance(p, str) or p not in _VARIANTS for p in selected):
        raise ValueError("profile_ids must contain known post-tuning profiles")
    if len(set(selected)) != len(selected):
        raise ValueError("profile_ids must be distinct")
    return selected


def reference_profile(profile_ids):
    editing = [p in (*EDIT_PROFILES, *_TRANSFER_VARIANTS, *_OBJECTIVE_VARIANTS) for p in profile_ids]
    if any(editing) and not all(editing):
        raise ValueError("Editing factorial and legacy profiles require separate plans")
    if any(p in _OBJECTIVE_VARIANTS for p in profile_ids):
        return 'transfer_relu'
    return "edit_control" if any(editing) else "baseline"


def needs_surrogate(config):
    return config.get("selector_rule") in GUIDED_SELECTOR_RULES or "editor_gain" in config


def build_trials(preset="quick", profile_ids=None, matched_controls=False):
    """Include baseline Post and either baseline or profile-matched controls.

    In matched mode, every selected profile runs on all three arms. If baseline
    was not selected, only its Post trial is added as the reference.
    """
    if type(matched_controls) is not bool:
        raise ValueError("matched_controls must be a boolean")
    selected = select_profiles(preset, profile_ids)
    anchor = reference_profile(selected)
    pairs = [(profile, "endpoint_post") for profile in selected]
    if anchor not in selected:
        pairs.insert(0, (anchor, "endpoint_post"))
    controls = selected if matched_controls else [anchor]
    pairs.extend((profile, arm) for profile in controls for arm in ARMS if arm != "endpoint_post")
    return [{"trial_id": f"{profile}__{arm}", "arm": arm,
             "profile_id": profile, "config": get_profile(profile)} for profile, arm in pairs]


def make_jobs(datasets, seeds, source_budget, max_actions, preset="quick", profile_ids=None,
              matched_controls=False):
    """Build independent trial jobs for a shared saved trajectory prefix."""
    datasets, seeds = _sequence(datasets, "datasets"), _sequence(seeds, "seeds")
    if (any(not isinstance(d, str) or d not in DATASETS for d in datasets)
            or len(set(datasets)) != len(datasets)):
        raise ValueError("datasets must be distinct supported dataset names")
    if (any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct nonnegative integers below 2**32")
    if (isinstance(source_budget, bool) or not isinstance(source_budget, Real)
            or not math.isfinite(source_budget) or not 0 < source_budget <= 1):
        raise ValueError("source_budget must be finite and in (0,1]")
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    trials = build_trials(preset, profile_ids, matched_controls)
    return [{**trial, "config": dict(trial["config"]), "dataset": dataset, "seed": seed,
             "source_budget": float(source_budget), "max_actions": max_actions}
            for dataset in datasets for seed in seeds for trial in trials]


def _finite_number(value, name, lower, upper=None, exclusive_lower=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < lower
            or (exclusive_lower and value == lower) or (upper is not None and value > upper)):
        raise ValueError(f"Invalid {name}: {value!r}")


def validate_config(config):
    """Validate a complete configuration, including caller-edited catalog copies.

    Unknown or missing fields fail closed. Probabilities and floors are bounded
    by one; visit exponents must be finite and nonnegative; all size limits are
    positive integers. The returned dictionary is a fresh copy.
    """
    mechanism_fields = {"selector_rule", "selector_pool_size"}
    editor_fields = {"editor_gain", "editor_gradient", "editor_candidates"}
    transfer_fields = {"editor_proxy", "editor_residual"}
    objective_fields = {'editor_objective', 'editor_local_mix'}
    if not isinstance(config, Mapping) or set(config) not in (set(_BASELINE), set(_BASELINE) | mechanism_fields,
            set(_BASELINE) | editor_fields, set(_BASELINE) | editor_fields | transfer_fields,
            set(_BASELINE) | editor_fields | transfer_fields | objective_fields):
        raise ValueError("Configuration must contain exactly the complete profile fields")
    if 'editor_objective' in config:
        if (config['editor_objective'] not in ('nll','capped_margin') or config['editor_proxy']!='relu'
                or config['editor_residual'] is not False):
            raise ValueError('Objective pilot requires ReLU full replacement and a known objective')
        _finite_number(config['editor_local_mix'], 'editor_local_mix', 0., 1.)
        if config['editor_local_mix'] == 1.:
            raise ValueError('Objective pilot must retain positive weight on every target')
    if 'editor_proxy' in config:
        if (config['editor_proxy'] not in ('source','relu') or type(config['editor_residual']) is not bool
                or config['editor_gradient'] is not True or config['editor_gain'] is not True):
            raise ValueError('Transfer editor requires a known proxy, boolean residual policy and both editing factors')
    if "editor_gain" in config:
        if type(config["editor_gain"]) is not bool or type(config["editor_gradient"]) is not bool or type(config["editor_candidates"]) is not int or config["editor_candidates"] != 3:
            raise ValueError("Editing requires boolean factors and exactly three candidates")
        if config["text_max_cluster_size"] != 4 or config["text_cdl_topk"] > config["text_max_added_words"]:
            raise ValueError("Editing requires a four-node group and feasible word caps")
    if "selector_rule" in config:
        if config["selector_rule"] not in (*_MECHANISMS, *COVERAGE_SELECTOR_RULES):
            raise ValueError("Unknown selector mechanism")
        if type(config["selector_pool_size"]) is not int or not 4 <= config["selector_pool_size"] <= 256:
            raise ValueError("selector_pool_size must be an integer in [4,256]")
        if config["readout"] != "last" or config["visit_gamma"] != 0:
            raise ValueError("Selector mechanisms require last-state readout and no visit penalty")
    if not isinstance(config["id"], str) or not re.fullmatch(r"[a-z0-9_]+", config["id"]):
        raise ValueError("Configuration id must be a safe lowercase identifier")
    for key in ("family", "description"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Configuration {key} must be a nonempty string")
    _finite_number(config["alpha"], "alpha", 0., 1., exclusive_lower=True)
    _finite_number(config["weight_floor"], "weight_floor", 0., 1.)
    _finite_number(config["visit_gamma"], "visit_gamma", 0.)
    _finite_number(config["text_similarity_min"], "text_similarity_min", 0., 1.)
    for key in ("iterations", "generation_max_tokens", "text_min_cluster_size",
                "text_max_cluster_size", "text_max_added_words", "text_cdl_topk", "text_budget_per_node"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Configuration {key} must be a positive integer")
    if config["text_min_cluster_size"] > config["text_max_cluster_size"]:
        raise ValueError("text_min_cluster_size must not exceed text_max_cluster_size")
    if config["readout"] not in ("last", "diffused"):
        raise ValueError("readout must be 'last' or 'diffused'")
    return dict(config)


def runtime_settings(config):
    """Apply configurable text options while preserving four writes and ten visits."""
    from gbhaa_experiments.action_state_protocol import protocol_settings
    config = validate_config(config)
    settings = protocol_settings()
    settings.update({key: value for key, value in config.items() if key.startswith("text_")})
    settings.update(text_ppr_alpha=config["alpha"], text_ppr_iters=config["iterations"])
    return settings


def operator_settings(config):
    """Describe the actual operator, including explicitly labeled method variants."""
    from gbhaa_experiments.action_state_protocol import operator_settings as base_settings
    config = validate_config(config)
    settings = base_settings()
    settings.update(
        local_restart=config["alpha"], local_updates=config["iterations"],
        local_readout="last_state" if config["readout"] == "last" else "max(last_state-alpha*q,0)",
        attribute_weights="floor+(1-floor)*clipped_cosine", weight_floor=config["weight_floor"],
        visit_gamma=config["visit_gamma"], ranking="readout/(1+visits)**visit_gamma",
        generation_max_tokens=config["generation_max_tokens"], scope="short_prefix_validation_development")
    if "selector_rule" in config:
        settings.update(selector_version=selector_version(config["selector_rule"]), ranking=config["selector_rule"],
                        selector_pool_size=config["selector_pool_size"],
                        selector_eligibility="visits<10 before candidate ranking",
                        selector_fallback="original diffusion order after positive non-restarts",
                        selector_surrogate="fixed clean model; fixed pseudo-labels; all non-training nodes"
                        if config["selector_rule"] in GUIDED_SELECTOR_RULES else "none")
        if config["selector_rule"] in GUIDED_SELECTOR_RULES:
            settings["selector_gradient_operator"] = "GCN on unweighted arm graph; self-loops and symmetric normalization"
        if config["selector_rule"] in COVERAGE_SELECTOR_RULES:
            quota = config["selector_rule"] == "frontier_editgain_fresh3"
            settings.update(selector_coverage_policy="reserve fresh quota, then raw gain" if quota else "gain/(1+prior visits)",
                            selector_fresh_quota=3 if quota else 0,
                            selector_quota_scope="original diffusion pool; no expansion; shortfall fills by existing ranking",
                            selector_zero_gain="retain fresh quota; diffusion/ID within groups" if quota else "diffusion/ID order")
    if "editor_gain" in config:
        settings.update(editor_version="projected-text-editor-v1", editor_candidates=3,
            editor_choice="projected_nll" if config["editor_gain"] else "seeded_random",
            editor_words="signed_gradient" if config["editor_gradient"] else "frequency",
            editor_graph="arm diffusion graph", editor_order="ascending node ID, sequential writes",
            editor_word_caps=[math.ceil(config["text_cdl_topk"]*(i+1)/3) for i in range(3)], editor_generation="three independent one-template requests",
            editor_surrogate="fixed clean GCN and pseudo-labels; all non-training nodes")
        if 'editor_proxy' in config:
            settings.update(editor_policy_version='transfer-editor-v1',editor_proxy=config['editor_proxy'],
                editor_edit_policy='planned_residual' if config['editor_residual'] else 'full_replacement',
                editor_proxy_fit='independent editor-transfer seed; CPU ReLU GCN; train labels only; Adam lr=.01 wd=.0005; 200 epochs; no validation'
                    if config['editor_proxy']=='relu' else 'existing clean structural surrogate',
                editor_pseudo_labels='unchanged source clean argmax; all non-training nodes')
        if 'editor_objective' in config:
            settings.update(editor_objective_version='arm-local-objective-v1',
                editor_objective=config['editor_objective'],editor_local_mix=config['editor_local_mix'],
                editor_choice='projected_objective',
                editor_objective_formula='-relu(logp[pseudo]-max_other_logp)' if config['editor_objective']=='capped_margin' else '-logp[pseudo]',
                editor_target_weights='(1-mix)/N + mix*arm_diffusion[target]/target_mass; uniform if mass<=1e-12',
                editor_weight_state='round-start features, arm graph and arm restarts; frozen for gradient and sequential candidate scoring')
    return settings


def _restart_distribution(restarts, n):
    restarts = _sequence(restarts, "restarts")
    if any(isinstance(u, (bool, np.bool_)) or not isinstance(u, Integral) or not 0 <= u < n
           for u in restarts):
        raise ValueError("restarts must contain valid integer node IDs")
    distinct = sorted({int(u) for u in restarts})
    q = np.zeros(n, dtype=np.float32)
    q[distinct] = 1. / len(distinct)
    return q


def diffusion_scores(features, adj, restarts, config):
    """Return float32 last-state diffusion scores on the supplied single graph.

    Restarts are uniform over distinct valid IDs (empty/invalid input fails).
    Existing non-self edges receive floor + (1-floor)*clipped_cosine, regardless
    of stored edge magnitude. Zero-weight rows receive self loops. With baseline
    settings this matches Heirattack.ppr_topk_from_seeds for distinct valid seeds.
    """
    config = validate_config(config)
    try:
        adjacency = sp.csr_matrix(adj, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("adj must be a numeric square matrix") from exc
    n = adjacency.shape[0]
    if (n == 0 or adjacency.shape != (n, n) or not np.isfinite(adjacency.data).all()
            or np.any(adjacency.data < 0)):
        raise ValueError("adj must be nonempty, square, finite and nonnegative")
    if torch.is_tensor(features):
        values = features.detach().to("cpu").float()
        if values.layout != torch.strided:
            values = values.to_dense()
    else:
        if sp.issparse(features):
            features = features.toarray()
        try:
            values = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("features must be a numeric node-by-feature matrix") from exc
    if (values.ndim != 2 or values.shape[0] != n or values.shape[1] == 0
            or not torch.isfinite(values).all()):
        raise ValueError("features must be a finite node-by-feature matrix matching adj")
    q = _restart_distribution(restarts, n)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    normalized = F.normalize(values, p=2, dim=1).numpy()
    rows, columns = adjacency.nonzero()
    cosine = np.clip((normalized[rows] * normalized[columns]).sum(axis=1), 0., 1.)
    floor = config["weight_floor"]
    weights = floor + (1. - floor) * cosine
    weighted = sp.csr_matrix((weights, (rows, columns)), shape=(n, n), dtype=np.float32)
    degree = np.asarray(weighted.sum(axis=1)).reshape(-1)
    isolates = degree < 1e-12
    if isolates.any():
        weighted = weighted.tolil()
        for node in np.where(isolates)[0]:
            weighted[node, node] = 1.
        weighted = weighted.tocsr()
        degree = np.asarray(weighted.sum(axis=1)).reshape(-1)
    transition = sp.diags(1. / np.maximum(degree, 1e-12), format="csr") @ weighted
    alpha = config["alpha"]
    state = q.copy()
    for _ in range(config["iterations"]):
        state = alpha * q + (1. - alpha) * (transition.T @ state)
    return np.asarray(state, dtype=np.float32)


def _visit_counts(visits, n):
    """Accept sparse node-ID/count mappings or dense length-n integer sequences."""
    counts = np.zeros(n, dtype=np.float64)
    if visits is None:
        return counts
    if isinstance(visits, Mapping):
        entries = visits.items()
    else:
        entries = enumerate(_sequence(visits, "visits"))
        entries = list(entries)
        if len(entries) != n:
            raise ValueError("Dense visits must have one count per node")
    for node, count in entries:
        if (isinstance(node, (bool, np.bool_)) or not isinstance(node, Integral)
                or not 0 <= node < n or isinstance(count, (bool, np.bool_))
                or not isinstance(count, Integral) or count < 0):
            raise ValueError("visits must contain valid integer IDs and nonnegative integer counts")
        try:
            counts[int(node)] = count
        except OverflowError as exc:
            raise ValueError("Visit counts must be finite") from exc
    if not np.isfinite(counts).all():
        raise ValueError("Visit counts must be finite")
    return counts


def score_nodes(features, adj, restarts, config, visits=None):
    """Return a full stable node order and float32 scores after readout/coverage.

    ``visits`` may be a node-ID -> count mapping (missing nodes mean zero), or a
    dense integer sequence of length n. Visits affect ranking only; diffusion
    and the arm's graph remain unchanged. Tied scores use ascending node IDs.
    """
    config = validate_config(config)
    if "selector_rule" in config:
        raise ValueError("Use selector_mechanisms for a mechanism configuration")
    # Materialize once so generators are usable for both diffusion and readout.
    restarts = _sequence(restarts, "restarts")
    scores = diffusion_scores(features, adj, restarts, config)
    if config["readout"] == "diffused":
        scores = np.maximum(scores - config["alpha"] * _restart_distribution(restarts, len(scores)), 0.)
    counts = _visit_counts(visits, len(scores))
    if config["visit_gamma"]:
        # Algebraically (1+visits)**-gamma, without overflow at large exponents.
        with np.errstate(over="ignore", under="ignore"):
            penalty = np.exp(-config["visit_gamma"] * np.log1p(counts))
        scores = (scores * penalty).astype(np.float32)
    return stable_order(scores), scores
