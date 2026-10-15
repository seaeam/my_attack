"""Read-only validation of matched action-state artifacts and recorded provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from gbhaa_experiments.action_state_protocol import (
    ARMS, PROTOCOL, actions_hash, feature_measures, flip_action, independent_restarts,
    make_jobs, operator_settings, protocol_settings, validate_trajectory,
)
from gbhaa_experiments.followup_protocol import array_hash, graph_hash, prediction_metrics, seed_stream
from gbhaa_experiments.run_followup import fingerprint
from gbhaa_experiments.summarize_followup import (
    _canonical, _hash, _load_graph, _load_predictions, _number, _object, _read_json, _require,
)

INPUT_HASHES = ("original_adj_hash", "original_features_hash", "training_nodes_hash",
                "target_nodes_hash", "evaluation_labels_hash")
COUNTERS = ("targets", "feature_writes", "feature_changes", "template_attempts",
            "template_successes", "template_failures", "node_failures", "cluster_failures", "cache_hits")


def _same(actual, expected, name):
    _require(_canonical(actual) == _canonical(expected), f"{name} mismatch")


def _identity(metadata, job):
    for key, value in job.items():
        _same(metadata.get(key), value, key)
    _same(metadata.get("protocol"), PROTOCOL, "protocol")
    _same(metadata.get("settings"), protocol_settings(), "settings")
    _same(metadata.get("operator_settings"), operator_settings(), "operator_settings")
    for key in (*INPUT_HASHES, "actions_hash", "modified_adj_hash"):
        _hash(metadata.get(key), key)
    _require(metadata.get("budget_complete") is True, "budget_complete must be true")
    for key in ("target_edges", "completed_edges"):
        _number(metadata.get(key), key, integer=True, minimum=1)
    _same(metadata["completed_edges"], metadata["target_edges"], "completed_edges")
    _number(metadata.get("structural_seconds"), "structural_seconds")


def _counts(value, writes, name):
    value = _object(value, name)
    for key in COUNTERS:
        _number(value.get(key), f"{name}.{key}", integer=True)
    for key in ("targets", "feature_writes"):
        _same(value[key], writes, f"{name}.{key}")
    _require(value["feature_changes"] <= writes, f"{name}.feature_changes exceeds writes")
    for key in ("template_failures", "node_failures", "cluster_failures"):
        _same(value[key], 0, f"{name}.{key}")
    _same(value["template_attempts"], value["template_successes"], f"{name}.template_attempts")
    return value


def _nodes(value, n, name, count=4):
    _require(isinstance(value, list) and len(value) == count and
             all(type(v) is int and 0 <= v < n for v in value) and len(set(value)) == count,
             f"{name} must contain {count} distinct valid nodes")
    return value


def _features(artifacts, metadata, n):
    try:
        with np.load(artifacts / "features.npz", allow_pickle=False) as data:
            original, modified = data["original"], data["modified"]
        _require(original.ndim == 2 and original.shape[0] == n and original.shape[1] > 0 and
                 original.dtype.kind in "fiu" and modified.dtype.kind in "fiu" and
                 np.isfinite(original).all() and np.isfinite(modified).all(), "Invalid features.npz arrays")
        _same(array_hash(original), metadata["original_features_hash"], "original_features_hash")
        _same(array_hash(modified), metadata.get("modified_features_hash"), "modified_features_hash")
        for key, value in feature_measures(original, modified).items():
            _number(metadata.get(key), key, integer=isinstance(value, int))
            _require(np.isclose(metadata[key], value, rtol=1e-10, atol=1e-10), f"{key} mismatch")
        with np.load(artifacts / "feature_updates.npz", allow_pickle=False) as data:
            nodes, values = data["nodes"], data["values"]
        b = metadata["target_edges"]
        _require(nodes.shape == (b, 4) and nodes.dtype.kind in "iu", "feature_updates.npz nodes shape/type invalid")
        _require(values.shape == (b, 4, original.shape[1]) and values.dtype == original.dtype and
                 values.dtype.kind in "fiu" and np.isfinite(values).all(), "feature_updates.npz values invalid")
        return original, modified, nodes, values
    except Exception as exc:
        raise ValueError(f"feature artifacts: {exc}") from exc


def _rounds(artifacts, metadata, original, features=None):
    rounds = _read_json(artifacts / "rounds.json").get("rounds")
    _require(isinstance(rounds, list) and len(rounds) == metadata["target_edges"], "rounds count mismatch")
    n = original.shape[0]
    state = original.copy()
    visits = np.zeros(n, dtype=np.int64)
    totals = dict.fromkeys(COUNTERS, 0)
    current = features[0].copy() if features is not None else None
    for i, (diagnostic, action) in enumerate(zip(rounds, metadata["actions"])):
        diagnostic = _object(diagnostic, f"round {i}")
        _same(diagnostic.get("round"), i, "round index")
        if "index" in diagnostic:
            _same(diagnostic["index"], i, "index")
        _same(diagnostic.get("action"), action, "round action")
        post = flip_action(state, action)
        _same(diagnostic.get("pre_graph_hash"), graph_hash(state), "pre_graph_hash")
        _same(diagnostic.get("post_graph_hash"), graph_hash(post), "post_graph_hash")
        if features is not None:
            local = metadata["local_graph"]
            _same(diagnostic.get("local_graph"), local, "local_graph")
            _same(diagnostic.get("diffusion_graph_hash"), graph_hash(post if local == "post" else state), "diffusion_graph_hash")
            _same(diagnostic.get("local_edge_value"), action[2] if local == "post" else 1-action[2], "local_edge_value")
            _same(diagnostic.get("grouping_graph"), "post", "grouping_graph")
            _same(diagnostic.get("grouping_graph_hash"), graph_hash(post), "grouping_graph_hash")
            restarts = _nodes(diagnostic.get("restart_nodes"), n, "restart_nodes", 2)
            expected = independent_restarts(n, metadata["seed"], i) if metadata["variant"] == "parallel" else action[:2]
            _same(sorted(restarts), sorted(expected), "restart_nodes")
            if metadata["variant"] == "parallel":
                _same(diagnostic.get("independent_restart_overlap"), len(set(restarts) & set(action[:2])), "independent_restart_overlap")
            else:
                _require(diagnostic.get("independent_restart_overlap") is None,
                         "independent_restart_overlap is only defined for Parallel")
            _same(diagnostic.get("candidate_universe"), n, "candidate_universe")
            eligible = visits < 10
            _same(diagnostic.get("eligible_count"), int(eligible.sum()), "eligible_count")
            selected = _nodes(diagnostic.get("selected_nodes"), n, "selected_nodes")
            pre = _nodes(diagnostic.get("probe_pre_nodes"), n, "probe_pre_nodes")
            after = _nodes(diagnostic.get("probe_post_nodes"), n, "probe_post_nodes")
            for nodes, name in ((selected, "selected_nodes"), (pre, "probe_pre_nodes"), (after, "probe_post_nodes")):
                _require(eligible[nodes].all(), f"{name} contains nodes exceeding max visits")
            if metadata["variant"] != "parallel":
                _same(selected, after if local == "post" else pre, "selected_nodes vs endpoint probe")
            _same(diagnostic.get("current_features_hash"), array_hash(current), "current_features_hash")
            overlap = len(set(pre) & set(after)) / len(set(pre) | set(after))
            _number(diagnostic.get("probe_jaccard"), "probe_jaccard")
            _require(abs(diagnostic["probe_jaccard"] - overlap) <= 1e-12, "probe_jaccard mismatch")
            _number(diagnostic.get("probe_score_l1"), "probe_score_l1")
            _number(diagnostic.get("action_similarity"), "action_similarity")
            _require(diagnostic["action_similarity"] <= 1 + 1e-6, "action_similarity exceeds 1")
            vectors = current[action[:2]]
            norm_product = np.prod(np.linalg.norm(vectors, axis=1))
            similarity = float(np.dot(*vectors) / norm_product) if norm_product > 1e-12 else 0.
            _require(abs(diagnostic["action_similarity"] - np.clip(similarity, 0, 1)) <= 1e-6,
                     "action_similarity differs from current features")
            counts = _counts(diagnostic.get("attribute_counts"), 4, f"round {i}.attribute_counts")
            for key in COUNTERS:
                totals[key] += counts[key]
            _same(features[2][i].tolist(), selected, "feature_updates.npz nodes vs selected_nodes")
            current[selected] = features[3][i]
            visits[selected] += 1
        state = post
    if features is not None:
        _require(np.array_equal(current, features[1]), "feature_updates.npz replay differs from final features")
        _same(metadata.get("attribute_node_visits"), int(visits.sum()), "attribute_node_visits")
        _same(metadata.get("visited_attribute_nodes"), int(np.count_nonzero(visits)), "visited_attribute_nodes")
        counts = _counts(metadata.get("attribute_counts"), 4*metadata["target_edges"], "attribute_counts")
        for key in COUNTERS:
            _same(counts[key], totals[key], f"attribute_counts.{key} total")


def _predictions(path, n):
    values = _load_predictions(path, n)
    try:
        with np.load(path, allow_pickle=False) as data:
            logits = data["log_probabilities"]
        _require(logits.ndim == 2 and logits.shape[0] == len(values["node_ids"]) and
                 logits.shape[1] > 1 and logits.dtype.kind == "f" and np.isfinite(logits).all(),
                 "log_probabilities must be a finite target-by-class array")
        _require(np.all(values["labels"] < logits.shape[1]), "prediction labels outside class count")
        _require(np.array_equal(logits.argmax(axis=1), values["predictions"]), "predictions disagree with log_probabilities")
    except Exception as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    return values


def _evaluation(artifacts, metadata, n):
    evaluation = _read_json(artifacts / "evaluation.json")
    _same(evaluation.get("protocol"), PROTOCOL, "evaluation.protocol")
    _same(evaluation.get("victim_seed"), seed_stream(metadata["seed"], "victim"), "victim_seed")
    metrics = _object(evaluation.get("metrics"), "evaluation.metrics")
    clean = _predictions(artifacts / "clean_predictions.npz", n)
    adv = _predictions(artifacts / "combined_predictions.npz", n)
    for key in ("node_ids", "labels"):
        _require(np.array_equal(clean[key], adv[key]), f"Prediction {key} mismatch")
    for key, field in (("node_ids", "target_nodes_hash"), ("labels", "evaluation_labels_hash")):
        _same(array_hash(clean[key]), metadata[field], field)
    _same(array_hash(clean["predictions"]), evaluation.get("clean_predictions_hash"), "clean_predictions_hash")
    exact = prediction_metrics(clean["predictions"], adv["predictions"], adv["labels"])
    exact["combined_accuracy"] = exact["attacked_accuracy"]
    for stage in ("edge", "feature"):
        key = f"{stage}_accuracy"
        _number(metrics.get(key), key)
        _require(metrics[key] <= 1, f"{key} exceeds 1")
        path = artifacts / f"{stage}_predictions.npz"
        if path.exists():
            predictions = _predictions(path, n)
            for field in ("node_ids", "labels"):
                _require(np.array_equal(clean[field], predictions[field]), f"{stage} prediction {field} mismatch")
            exact[key] = float((predictions["predictions"] == clean["labels"]).mean())
    for key, value in exact.items():
        if value is None:
            _require(key in metrics and metrics[key] is None, f"{key} must be undefined")
        else:
            actual = _number(metrics.get(key), key, integer=type(value) is int, minimum=-100 if key == "accuracy_drop_pp" else 0)
            _require(abs(actual - value) <= 1e-10, f"{key} disagrees with prediction artifacts")
    return evaluation


def load_stage(run_dir: Path, job: dict, trajectory_dir: Path | None = None) -> dict:
    """Validate artifacts before completion, or verify a completed saved stage."""
    run_dir = Path(run_dir)
    _require(job in make_jobs([job["dataset"]], [job["seed"]], [job["budget"]]), "Invalid job")
    structure = job["variant"] == "structure"
    artifacts = run_dir / "artifacts"
    name = "trajectory" if structure else "attack"
    metadata = _read_json(artifacts / f"{name}.json")
    _identity(metadata, job)
    original = _load_graph(artifacts / "original_adj.npz")
    modified = _load_graph(artifacts / "modified_adj.npz")
    _same(graph_hash(original), metadata["original_adj_hash"], "original_adj_hash")
    _same(graph_hash(modified), metadata["modified_adj_hash"], "modified_adj_hash")
    _same(metadata["target_edges"], int((original.nnz // 2)*job["budget"]), "target_edges")
    actions = metadata.get("actions")
    _require(isinstance(actions, list), "actions must be a list")
    _same(actions_hash(actions), metadata["actions_hash"], "actions_hash")
    validate_trajectory(original, modified, actions, metadata["target_edges"])
    result = {name: metadata}
    if structure:
        _rounds(artifacts, metadata, original)
    else:
        trajectory_dir = Path(trajectory_dir) if trajectory_dir is not None else run_dir.parent / "structure"
        trajectory = load_stage(trajectory_dir, {**job, "variant": "structure"})["trajectory"]
        for key in (*INPUT_HASHES, "actions_hash", "actions", "modified_adj_hash", "target_edges"):
            _same(metadata[key], trajectory[key], f"shared trajectory {key}")
        _same(metadata.get("trajectory_hash"), trajectory["actions_hash"], "trajectory_hash")
        _same(metadata.get("selector"), "ae_pr", "selector")
        _same(metadata.get("global_source"), "degree", "global_source")
        _same(metadata.get("local_graph"), "post" if job["variant"] == "endpoint_post" else "pre", "local_graph")
        _same(metadata.get("generation_seed"), seed_stream(job["seed"], "generation"), "generation_seed")
        _number(metadata.get("generation_temperature"), "generation_temperature")
        _require(metadata["generation_temperature"] == 0, "generation_temperature must be 0")
        _require(isinstance(metadata.get("generation_model"), str) and bool(metadata["generation_model"].strip()), "generation_model missing")
        for field in ("vocabulary_fallback", "vocabulary_partial"):
            _require(type(metadata.get(field)) is bool, f"{field} must be boolean")
        for field in ("attribute_node_visits", "visited_attribute_nodes"):
            _number(metadata.get(field), field, integer=True)
        cache = _object(metadata.get("request_cache"), "request_cache")
        for field in ("hits", "misses"):
            _number(cache.get(field), f"request_cache.{field}", integer=True)
        keys = metadata.get("generation_request_keys")
        _require(isinstance(keys, list) and all(isinstance(v, str) for v in keys) and len(set(keys)) == len(keys), "generation_request_keys must be distinct")
        for key in keys:
            _hash(key, "generation_request_keys digest")
        _require(len(keys) <= cache["hits"] + cache["misses"], "generation_request_keys exceed requests")
        _require(cache["misses"] <= len(keys) and bool(keys) == bool(cache["hits"] + cache["misses"]),
                 "generation_request_keys inconsistent with request_cache counters")
        _rounds(artifacts, metadata, original, _features(artifacts, metadata, original.shape[0]))
        result["evaluation"] = _evaluation(artifacts, metadata, original.shape[0])
    result_path = run_dir / "result.json"
    if result_path.exists():
        saved = _read_json(result_path)
        if saved.get("status") == "complete":
            for key, value in result.items():
                _same(saved.get(key), value, f"{key}.json vs result.{key}")
    return result


def _generation_cache(result_path, attack, provenance):
    """Verify every referenced request/reply in the saved provenance's cache."""
    cache_root = result_path.parents[4] / "_generation_cache" / fingerprint(provenance)
    for key in attack["generation_request_keys"]:
        path = cache_root / f"{key}.json"
        try:
            record = _read_json(path)
            _same(record.get("key"), key, "key")
            request = _object(record.get("request"), "request")
            encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
            _same(hashlib.sha256(encoded.encode()).hexdigest(), key, "request digest")
            _same(request.get("model"), attack["generation_model"], "request.model")
            _same(request.get("seed"), attack["generation_seed"], "request.seed")
            _number(request.get("temperature"), "request.temperature")
            _require(request["temperature"] == attack["generation_temperature"], "request.temperature mismatch")
            # v1 uses TextAttackGenerator's fixed 50-token default. Freeze it here:
            # importing a future generator default would invalidate historical runs.
            _number(request.get("max_tokens"), "request.max_tokens", integer=True, minimum=1)
            _same(request["max_tokens"], 50, "request.max_tokens")
            messages = request.get("messages")
            _require(isinstance(messages, list) and bool(messages) and
                     all(isinstance(message, dict) and isinstance(message.get("role"), str) and
                         bool(message["role"].strip()) and isinstance(message.get("content"), str) and
                         bool(message["content"].strip()) for message in messages),
                     "request.messages must contain nonempty role/content messages")
            reply = record.get("response")
            _require(isinstance(reply, str) and any(template.strip() for template in reply.split("|||")),
                     "response contains no parsed templates")
            _same(record.get("response_hash"), hashlib.sha256(reply.encode()).hexdigest(), "response_hash")
        except Exception as exc:
            raise ValueError(f"generation cache {path}: {exc}") from exc


def validate_result(path: Path, job: dict) -> dict:
    """Verify recorded fingerprints internally; historical source stays analyzable."""
    path = Path(path)
    row = _read_json(path)
    for key, value in job.items():
        _same(row.get(key), value, key)
    _same(row.get("protocol"), PROTOCOL, "protocol")
    _same(row.get("settings"), protocol_settings(), "settings")
    _same(row.get("status"), "complete", "status")
    _same(row.get("returncode"), 0, "returncode")
    _number(row.get("wall_time_seconds"), "wall_time_seconds")
    _require("peak_rss_mib" in row, "peak_rss_mib missing")
    if row["peak_rss_mib"] is not None:
        _number(row["peak_rss_mib"], "peak_rss_mib")
    command = row.get("command")
    _require(isinstance(command, list) and bool(command) and all(isinstance(v, str) and v for v in command), "command must be a nonempty argument list")
    provenance = _object(row.get("provenance"), "provenance")
    _same(provenance.get("protocol"), PROTOCOL, "provenance.protocol")
    for key in ("sources", "input_files"):
        manifest = _object(provenance.get(key), f"provenance.{key}")
        _require(bool(manifest), f"provenance.{key} must be nonempty")
        for name, digest in manifest.items():
            _require(isinstance(name, str) and bool(name.strip()), f"Invalid {key} path")
            _hash(digest, f"{key} digest")
    for key in ("python_executable", "api_base_url", "generation_model", "machine"):
        _require(isinstance(provenance.get(key), str) and bool(provenance[key].strip()), f"Missing provenance.{key}")
    _require(bool(_object(provenance.get("python_environment"), "python_environment")), "python_environment must be nonempty")
    _require("cuda_visible_devices" in provenance and
             (provenance["cuda_visible_devices"] is None or isinstance(provenance["cuda_visible_devices"], str)), "cuda_visible_devices missing/invalid")
    _same(provenance.get("thread_count"), 1, "thread_count")
    _number(provenance.get("generation_temperature"), "generation_temperature")
    _require(provenance["generation_temperature"] == 0, "generation_temperature must be 0")
    stage = load_stage(path.parent, job)
    trajectory_hash = None if job["variant"] == "structure" else stage["attack"]["trajectory_hash"]
    expected = fingerprint(dict(job=job, settings=protocol_settings(), provenance=provenance, trajectory_hash=trajectory_hash))
    _same(row.get("fingerprint"), expected, "fingerprint")
    if job["variant"] in ARMS:
        _same(stage["attack"]["generation_model"], provenance["generation_model"], "generation_model vs provenance")
        _generation_cache(path, stage["attack"], provenance)
    return row
