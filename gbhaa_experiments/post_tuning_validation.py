"""Read-only audits for validation-only, short-prefix Post development trials."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from gbhaa_experiments import action_state_validation as parent_validation
from gbhaa_experiments.action_state_protocol import PROTOCOL as PARENT_PROTOCOL
from gbhaa_experiments.action_state_protocol import actions_hash, validate_trajectory
from gbhaa_experiments.action_state_validation import INPUT_HASHES, _features, _predictions, _rounds, _same
from gbhaa_experiments.followup_protocol import array_hash, graph_hash, seed_stream
from gbhaa_experiments.post_tuning_protocol import PROTOCOL, make_jobs, operator_settings, runtime_settings
from gbhaa_experiments.run_followup import fingerprint
from gbhaa_experiments.summarize_followup import _hash, _load_graph, _number, _object, _read_json, _require

JOB_FIELDS = ("dataset", "seed", "source_budget", "max_actions", "trial_id", "arm", "profile_id", "config")
METRICS = ("clean_validation_accuracy", "validation_accuracy", "clean_validation_nll",
           "validation_nll", "accuracy_drop_pp")


def validate_job(job):
    """Require a concrete catalog configuration, even when reading a saved plan."""
    _object(job, "job")
    _same(sorted(job), sorted(JOB_FIELDS), "job fields")
    possible = make_jobs([job["dataset"]], [job["seed"]], job["source_budget"], job["max_actions"],
                         profile_ids=[job["profile_id"]], matched_controls=True)
    _require(job in possible, "job config/profile/arm differs from the saved catalog identity")


def _provenance(value):
    value = _object(value, "provenance")
    _same(value.get("protocol"), PROTOCOL, "provenance.protocol")
    for key in ("sources", "input_files"):
        manifest = _object(value.get(key), f"provenance.{key}")
        _require(bool(manifest), f"provenance.{key} must be nonempty")
        for name, digest in manifest.items():
            _require(isinstance(name, str) and bool(name.strip()), f"Invalid {key} path")
            _hash(digest, f"{key} digest")
    for key in ("python_executable", "api_base_url", "generation_model", "machine"):
        _require(isinstance(value.get(key), str) and bool(value[key].strip()), f"Missing provenance.{key}")
    _require(bool(_object(value.get("python_environment"), "python_environment")), "python_environment must be nonempty")
    _require("cuda_visible_devices" in value and
             (value["cuda_visible_devices"] is None or isinstance(value["cuda_visible_devices"], str)),
             "cuda_visible_devices missing/invalid")
    _same(value.get("thread_count"), 1, "thread_count")
    _number(value.get("generation_temperature"), "generation_temperature")
    _require(value["generation_temperature"] == 0, "generation_temperature must be 0")
    return value


def _splits(artifacts, metadata, n):
    """Read node IDs only: held-out labels/predictions are never loaded."""
    with np.load(artifacts / "splits.npz", allow_pickle=False) as data:
        splits = {key: data[key] for key in ("train", "validation", "test")}
    for key, values in splits.items():
        _require(values.ndim == 1 and len(values) > 0 and values.dtype.kind in "iu" and
                 np.all((values >= 0) & (values < n)) and len(np.unique(values)) == len(values),
                 f"{key} split must contain distinct nonempty valid node IDs")
    joined = np.concatenate(list(splits.values()))
    _require(len(np.unique(joined)) == len(joined), "train/validation/test splits must be disjoint")
    for key, field in (("train", "training_nodes_hash"), ("validation", "validation_nodes_hash"),
                       ("test", "holdout_nodes_hash")):
        _same(array_hash(splits[key]), metadata.get(field), field)
    targets = np.sort(np.concatenate((splits["validation"], splits["test"])))
    _same(array_hash(targets), metadata.get("target_nodes_hash"), "target_nodes_hash")
    return splits


def _validation_predictions(path, n):
    values = _predictions(path, n)
    with np.load(path, allow_pickle=False) as data:
        log_probabilities = data["log_probabilities"]
    _require(np.all(log_probabilities <= 1e-6) and
             np.allclose(np.exp(log_probabilities).sum(axis=1), 1., rtol=1e-5, atol=1e-5),
             f"{path.name}: log_probabilities are not normalized")
    values["log_probabilities"] = log_probabilities
    return values


def _evaluation(artifacts, metadata, splits, n):
    evaluation = _read_json(artifacts / "evaluation.json")
    _same(sorted(evaluation), sorted(("protocol", "split", "victim_seed", "validation_nodes_hash",
        "holdout_nodes_hash", "train_nodes_hash", "clean_predictions_hash", "validation_predictions_hash",
        "target_count", "metrics")), "validation-only evaluation fields")
    _same(evaluation.get("protocol"), PROTOCOL, "evaluation.protocol")
    _same(evaluation.get("split"), "validation", "evaluation.split")
    _same(evaluation.get("victim_seed"), seed_stream(metadata["seed"], "victim"), "victim_seed")
    for key, field in (("train_nodes_hash", "training_nodes_hash"), ("validation_nodes_hash", "validation_nodes_hash"),
                       ("holdout_nodes_hash", "holdout_nodes_hash")):
        _same(evaluation.get(key), metadata[field], f"evaluation.{key}")
    clean = _validation_predictions(artifacts / "clean_validation_predictions.npz", n)
    adv = _validation_predictions(artifacts / "validation_predictions.npz", n)
    for key in ("node_ids", "labels"):
        _require(np.array_equal(clean[key], adv[key]), f"Validation prediction {key} mismatch")
    _require(np.array_equal(clean["node_ids"], splits["validation"]), "prediction node_ids differ from validation split")
    _same(evaluation.get("target_count"), len(splits["validation"]), "target_count")
    for values, field in ((clean, "clean_predictions_hash"), (adv, "validation_predictions_hash")):
        _same(evaluation.get(field), array_hash(values["predictions"]), field)
    exact = {}
    for values, prefix in ((clean, "clean_validation"), (adv, "validation")):
        exact[f"{prefix}_accuracy"] = float((values["predictions"] == values["labels"]).mean())
        exact[f"{prefix}_nll"] = float(-values["log_probabilities"][np.arange(len(values["labels"])), values["labels"]].astype(np.float64).mean())
    exact["accuracy_drop_pp"] = 100 * (exact["clean_validation_accuracy"] - exact["validation_accuracy"])
    metrics = _object(evaluation.get("metrics"), "evaluation.metrics")
    _same(sorted(metrics), sorted(METRICS), "validation-only metrics")
    for key, value in exact.items():
        actual = _number(metrics.get(key), key, minimum=-100 if key == "accuracy_drop_pp" else 0)
        _require(abs(actual - value) <= 1e-10, f"{key} disagrees with prediction artifacts")
    return evaluation


def load_trial(run_dir: Path, job: dict, parent_run: Path | None = None) -> dict:
    """Audit artifacts before completion, optionally comparing a saved result snapshot."""
    validate_job(job)
    run_dir = Path(run_dir)
    artifacts = run_dir / "artifacts"
    metadata = _read_json(artifacts / "attack.json")
    for key in ("dataset", "seed", "source_budget", "profile_id", "trial_id", "config"):
        _same(metadata.get(key), job[key], key)
    for key in ("arm", "max_actions"):
        if key in metadata:
            _same(metadata[key], job[key], key)
    _same(metadata.get("variant"), job["arm"], "variant")
    _same(metadata.get("protocol"), PROTOCOL, "protocol")
    _same(metadata.get("settings"), runtime_settings(job["config"]), "settings")
    _same(metadata.get("operator_settings"), operator_settings(job["config"]), "operator_settings")
    _require(metadata.get("budget_complete") is True, "budget_complete must be true")
    for key in ("prefix_length", "target_edges", "completed_edges"):
        _same(metadata.get(key), job["max_actions"], key)
    _number(metadata.get("structural_seconds"), "structural_seconds")
    for field in (*INPUT_HASHES, "actions_hash", "trajectory_hash", "modified_adj_hash", "parent_fingerprint", "parent_actions_hash"):
        _hash(metadata.get(field), field)
    result_path = run_dir / "result.json"
    saved = _read_json(result_path) if result_path.exists() else None
    if parent_run is None:
        _require(saved is not None and isinstance(saved.get("parent_run"), str), "parent_run required")
        parent_run = Path(saved["parent_run"])
    parent_run = Path(parent_run)
    _require(parent_run.is_absolute(), "parent_run must be an absolute original structure directory")
    parent_job = dict(dataset=job["dataset"], seed=job["seed"], budget=job["source_budget"], variant="structure")
    parent = parent_validation.validate_result(parent_run / "result.json", parent_job)
    trajectory = parent["trajectory"]
    _same(metadata.get("parent_protocol"), PARENT_PROTOCOL, "parent_protocol")
    _same(metadata["parent_fingerprint"], parent["fingerprint"], "parent_fingerprint")
    _same(metadata["parent_actions_hash"], trajectory["actions_hash"], "parent_actions_hash")
    _require(job["max_actions"] <= trajectory["target_edges"], "prefix exceeds parent trajectory")
    expected_actions = trajectory["actions"][:job["max_actions"]]
    _same(metadata.get("actions"), expected_actions, "actions must be the ordered parent prefix")
    for key in INPUT_HASHES:
        _same(metadata[key], trajectory[key], f"parent {key}")
    for key in ("actions_hash", "trajectory_hash"):
        _same(metadata[key], actions_hash(expected_actions), key)
    original = _load_graph(artifacts / "original_adj.npz")
    modified = _load_graph(artifacts / "modified_adj.npz")
    _same(graph_hash(original), metadata["original_adj_hash"], "original_adj_hash")
    _same(graph_hash(modified), metadata["modified_adj_hash"], "modified_adj_hash")
    edges = original.nnz // 2
    _require(edges > 0, "original graph has no edges")
    for key in ("budget", "effective_budget"):
        _number(metadata.get(key), key)
        _require(np.isclose(metadata[key], job["max_actions"] / edges, rtol=1e-12, atol=1e-12), f"{key} differs from actual prefix budget")
    validate_trajectory(original, modified, expected_actions, job["max_actions"])
    _same(metadata.get("selector"), "ae_pr", "selector")
    _same(metadata.get("global_source"), "degree", "global_source")
    _same(metadata.get("local_graph"), "post" if job["arm"] == "endpoint_post" else "pre", "local_graph")
    _same(metadata.get("generation_seed"), seed_stream(job["seed"], "generation"), "generation_seed")
    _number(metadata.get("generation_temperature"), "generation_temperature")
    _require(metadata["generation_temperature"] == 0, "generation_temperature must be 0")
    _same(metadata.get("generation_max_tokens"), job["config"]["generation_max_tokens"], "generation_max_tokens")
    _require(isinstance(metadata.get("generation_model"), str) and bool(metadata["generation_model"].strip()), "generation_model missing")
    for key in ("vocabulary_fallback", "vocabulary_partial"):
        _require(type(metadata.get(key)) is bool, f"{key} must be boolean")
    cache = _object(metadata.get("request_cache"), "request_cache")
    for key in ("hits", "misses"):
        _number(cache.get(key), f"request_cache.{key}", integer=True)
    keys = metadata.get("generation_request_keys")
    _require(isinstance(keys, list) and all(isinstance(key, str) for key in keys) and len(keys) == len(set(keys)), "generation_request_keys must be distinct")
    for key in keys:
        _hash(key, "generation_request_keys digest")
    _require(cache["misses"] <= len(keys) <= cache["hits"] + cache["misses"] and
             bool(keys) == bool(cache["hits"] + cache["misses"]), "generation_request_keys inconsistent with cache counters")
    feature_artifacts = _features(artifacts, metadata, original.shape[0])
    _rounds(artifacts, metadata, original, feature_artifacts)
    splits = _splits(artifacts, metadata, original.shape[0])
    if "selector_rule" in job["config"]:
        from gbhaa_experiments.selector_mechanism_validation import audit_mechanism_rounds
        audit_mechanism_rounds(artifacts, metadata, original, feature_artifacts, splits)
    if "editor_gain" in job["config"]:
        from gbhaa_experiments.projected_editing_validation import audit_editor
        audit_editor(artifacts, metadata, original, feature_artifacts, splits)
    result = dict(attack=metadata, evaluation=_evaluation(artifacts, metadata, splits, original.shape[0]))
    if saved is not None and saved.get("status") == "complete":
        for key, value in result.items():
            _same(saved.get(key), value, f"{key}.json vs result.{key}")
        _same(saved.get("parent_fingerprint"), parent["fingerprint"], "result.parent_fingerprint")
        _same(saved.get("parent_run"), str(parent_run), "result.parent_run")
    return result


def _generation_cache(result_path, attack, provenance):
    cache_root = result_path.parents[5] / "_generation_cache" / fingerprint(provenance)
    editor_records = {}
    if "editor_gain" in attack["config"]:
        for diagnostic in _read_json(result_path.parent / "artifacts/rounds.json")["rounds"]:
            for record in diagnostic["editing"]["requests"]:
                if record["key"] in editor_records:
                    _same(editor_records[record["key"]], record, "repeated editor request record")
                editor_records[record["key"]] = record
        _same(sorted(editor_records), sorted(attack["generation_request_keys"]), "editor cache keys")
    for key in attack["generation_request_keys"]:
        path = cache_root / f"{key}.json"
        try:
            record = _read_json(path)
            if editor_records:
                _same(record, editor_records[key], "editor response vs persistent cache")
            _same(record.get("key"), key, "key")
            request = _object(record.get("request"), "request")
            encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
            _same(hashlib.sha256(encoded.encode()).hexdigest(), key, "request digest")
            for key_name, field in (("model", "generation_model"), ("seed", "generation_seed"),
                                    ("max_tokens", "generation_max_tokens")):
                _same(request.get(key_name), attack[field], f"request.{key_name}")
            _number(request.get("temperature"), "request.temperature")
            _require(request["temperature"] == attack["generation_temperature"], "request.temperature mismatch")
            messages = request.get("messages")
            _require(isinstance(messages, list) and bool(messages) and
                     all(isinstance(message, dict) and isinstance(message.get("role"), str) and
                         bool(message["role"].strip()) and isinstance(message.get("content"), str) and
                         bool(message["content"].strip()) for message in messages), "request.messages invalid")
            response = record.get("response")
            _require(isinstance(response, str) and any(part.strip() for part in response.split("|||")), "response contains no parsed templates")
            _same(record.get("response_hash"), hashlib.sha256(response.encode()).hexdigest(), "response_hash")
        except Exception as exc:
            raise ValueError(f"generation cache {path}: {exc}") from exc


def validate_result(path: Path, job: dict) -> dict:
    """Validate saved identity and artifacts without comparing to today's checkout."""
    validate_job(job)
    path = Path(path)
    row = _read_json(path)
    for key, value in job.items():
        _same(row.get(key), value, key)
    _same(row.get("protocol"), PROTOCOL, "protocol")
    _same(row.get("settings"), runtime_settings(job["config"]), "settings")
    _same(row.get("status"), "complete", "status")
    _same(row.get("returncode"), 0, "returncode")
    _number(row.get("wall_time_seconds"), "wall_time_seconds")
    _require("peak_rss_mib" in row, "peak_rss_mib missing")
    if row["peak_rss_mib"] is not None:
        _number(row["peak_rss_mib"], "peak_rss_mib")
    command = row.get("command")
    _require(isinstance(command, list) and bool(command) and all(isinstance(v, str) and v for v in command), "command must be a nonempty argument list")
    provenance = _provenance(row.get("provenance"))
    _hash(row.get("parent_fingerprint"), "parent_fingerprint")
    expected = fingerprint(dict(job=job, provenance=provenance, parent_fingerprint=row["parent_fingerprint"]))
    _same(row.get("fingerprint"), expected, "fingerprint")
    stage = load_trial(path.parent, job)
    _same(stage["attack"]["generation_model"], provenance["generation_model"], "generation_model vs provenance")
    _generation_cache(path, stage["attack"], provenance)
    return row
