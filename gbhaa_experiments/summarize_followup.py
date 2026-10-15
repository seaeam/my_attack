"""Paired, complete-block summaries. Never inserts numbers into a manuscript."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np
import scipy.sparse as sp
from scipy.stats import t

from gbhaa_experiments.run_followup import (DEFAULT_OUTPUT, PROTOCOL, SUITES,
                                           fingerprint, make_jobs, settings,
                                           default_output, protocol_for_suite)
from gbhaa_experiments.followup_config import FACTORIAL_PROTOCOL, pr_solver_settings, factorial_operator_settings
from gbhaa_experiments.runner import write_json
from gbhaa_experiments.followup_protocol import (
    array_hash, graph_hash, prediction_metrics, seed_stream,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(value, name):
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _number(value, name, *, integer=False, minimum=0):
    valid = type(value) is int if integer else type(value) in (int, float)
    _require(valid and math.isfinite(value) and value >= minimum,
             f"{name} must be a finite {'integer' if integer else 'number'} >= {minimum}")
    return value


def _hash(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             f"{name} must be a SHA-256 digest")


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        # Also reject NaN/Infinity anywhere, including optional metadata.
        try:
            _canonical(value)
        except ValueError as exc:
            raise ValueError("non-finite JSON value") from exc
        return _object(value, path.name)
    except Exception as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def _validate_metadata(row, job):
    """Validate recorded identity, not the current source/input file contents."""
    _object(row, "result")
    for key, value in job.items():
        _require(_canonical(row.get(key)) == _canonical(value),
                 f"Stored {key} differs from directory/job")
    _require(row.get("status") == "complete", "Run incomplete/failed")
    protocol = protocol_for_suite(job["suite"])
    _require(row.get("protocol") == protocol, "Invalid result protocol")
    _require(type(row.get("returncode")) is int and row["returncode"] == 0,
             "returncode must be 0")
    _require(_canonical(row.get("settings")) == _canonical(settings(job)),
             "settings differ from protocol")
    provenance = _object(row.get("provenance"), "provenance")
    _require(provenance.get("protocol") == protocol, "Invalid provenance protocol")
    for key in ("sources", "input_files"):
        manifest = _object(provenance.get(key), "provenance." + key)
        _require(bool(manifest), f"provenance.{key} must be nonempty")
        for name, digest in manifest.items():
            _require(isinstance(name, str) and bool(name.strip()), f"Invalid {key} path")
            _hash(digest, f"{key} digest")
    for key in ("python_executable", "api_base_url", "generation_model", "machine"):
        _require(isinstance(provenance.get(key), str) and bool(provenance[key].strip()),
                 f"Missing/invalid provenance.{key}")
    _require(bool(_object(provenance.get("python_environment"), "python_environment")),
             "python_environment must be nonempty")
    _require("cuda_visible_devices" in provenance and
             (provenance["cuda_visible_devices"] is None or isinstance(provenance["cuda_visible_devices"], str)),
             "Missing/invalid cuda_visible_devices")
    _number(provenance.get("thread_count"), "thread_count", integer=True, minimum=1)
    _number(provenance.get("generation_temperature"), "generation_temperature")
    _require(provenance["generation_temperature"] == 0, "generation_temperature must be 0")
    expected = fingerprint(dict(job=job, settings=row["settings"], provenance=provenance))
    _require(row.get("fingerprint") == expected, "fingerprint mismatch")

    attack = _object(row.get("attack"), "attack")
    evaluation = _object(row.get("evaluation"), "evaluation")
    _object(evaluation.get("metrics"), "evaluation.metrics")
    for section, name in ((attack, "attack"), (evaluation, "evaluation")):
        _require(section.get("protocol") == protocol, f"Invalid {name}.protocol")
    if job["selector"] == "pr":
        _require(attack.get("pr_solver") == pr_solver_settings(row["settings"]["text_ppr_alpha"]),
                 "pr_solver differs from the ordinary PageRank configuration")
    if protocol == FACTORIAL_PROTOCOL:
        _require(attack.get("operator_settings") == factorial_operator_settings(),
                 "operator_settings differ from the factorial configuration")
        _require(attack.get("requested_local_graph") == "pre", "requested_local_graph must be pre")
    for key in ("seed", "dataset", "selector", "local_graph", "global_transition", "global_source"):
        _require(_canonical(attack.get(key)) == _canonical(job[key]), f"attack.{key} differs from job")
    for section, key, stream in ((attack, "generation_seed", "generation"),
                                 (evaluation, "victim_seed", "victim")):
        _require(type(section.get(key)) is int and section[key] == seed_stream(job["seed"], stream),
                 f"{key} differs from expected seed stream")
    _number(attack.get("generation_temperature"), "attack.generation_temperature")
    _require(attack["generation_temperature"] == 0, "attack.generation_temperature must be 0")
    _require(attack.get("generation_model") == provenance["generation_model"],
             "attack.generation_model differs from provenance")
    for key in ("original_adj_hash", "original_features_hash", "training_nodes_hash",
                "target_nodes_hash", "evaluation_labels_hash", "actions_hash", "modified_adj_hash"):
        _hash(attack.get(key), "attack." + key)
    _hash(evaluation.get("clean_predictions_hash"), "clean_predictions_hash")
    _require(attack.get("budget_complete") is True, "edge budget incomplete")
    if "edge_budget_complete" in attack:
        _require(attack["edge_budget_complete"] is True, "edge_budget_complete is false")
    for key in ("target_edges", "completed_edges", "changed_feature_nodes",
                "changed_feature_entries", "attribute_node_visits"):
        _number(attack.get(key), "attack." + key, integer=True)
    _require(attack["completed_edges"] == attack["target_edges"], "edge budget incomplete")
    _number(attack.get("structural_seconds"), "structural_seconds")
    _number(row.get("wall_time_seconds"), "wall_time_seconds")
    _require("peak_rss_mib" in row, "Missing peak_rss_mib")
    if row["peak_rss_mib"] is not None:
        _number(row["peak_rss_mib"], "peak_rss_mib")


def _load_graph(path):
    try:
        graph = sp.load_npz(path).tocsr(copy=True)
        graph.sum_duplicates()
        graph.eliminate_zeros()
        graph.sort_indices()
        _require(graph.shape[0] == graph.shape[1] and graph.shape[0] > 0,
                 "adjacency must be nonempty and square")
        _require(np.isfinite(graph.data).all() and np.all(graph.data == 1),
                 "adjacency must be binary")
        _require((graph != graph.T).nnz == 0, "adjacency must be symmetric")
        _require(not np.any(graph.diagonal()), "adjacency must have no self loops")
        return graph
    except Exception as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def _validate_actions(attack, original, modified, budget):
    _require(graph_hash(original) == attack["original_adj_hash"], "original_adj_hash mismatch")
    _require(graph_hash(modified) == attack["modified_adj_hash"], "modified_adj_hash mismatch")
    _require(original.shape == modified.shape, "modified adjacency shape mismatch")
    target = int((original.nnz // 2) * budget)
    _require(attack["target_edges"] == target, "target edge budget differs from original graph/job")
    actions = attack.get("actions")
    _require(isinstance(actions, list) and len(actions) == target, "actions count differs from edge budget")
    digest = hashlib.sha256(json.dumps(actions).encode()).hexdigest()
    _require(digest == attack["actions_hash"], "actions_hash mismatch")
    replay = original.tolil(copy=True)
    seen = set()
    for action in actions:
        _require(isinstance(action, list) and len(action) == 3 and
                 all(type(value) is int for value in action), "actions must be integer [u, v, value] triples")
        u, v, value = action
        _require(0 <= u < original.shape[0] and 0 <= v < original.shape[0], "action endpoint out of range")
        _require(u != v, "action self loop")
        edge = (min(u, v), max(u, v))
        _require(edge not in seen, "action reflip of an already changed edge")
        seen.add(edge)
        _require(value in (0, 1) and value == 1 - replay[u, v], "action does not flip its edge")
        replay[u, v] = replay[v, u] = value
    _require((replay.tocsr() != modified).nnz == 0, "action replay differs from modified adjacency")
    _require((original != modified).nnz // 2 == attack["completed_edges"], "completed_edges differs from graph")


def _load_predictions(path, node_count):
    try:
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in ("node_ids", "labels", "predictions")}
        shape = values["node_ids"].shape
        _require(len(shape) == 1 and shape[0] > 0 and all(v.shape == shape for v in values.values()),
                 "prediction arrays must have the same nonempty 1-D shape")
        for key, value in values.items():
            _require(value.dtype.kind in "iu" and np.all(value >= 0), f"{key} must contain nonnegative integers")
        ids = values["node_ids"]
        _require(len(np.unique(ids)) == len(ids), "node_ids must be unique")
        _require(np.all(ids < node_count), "node_ids outside graph")
        return values
    except Exception as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def _read_validated_record(path, job):
    row = _read_json(path)
    _validate_metadata(row, job)
    artifacts = path.parent / "artifacts"
    for name in ("attack", "evaluation"):
        saved = _read_json(artifacts / f"{name}.json")
        _require(_canonical(saved) == _canonical(row[name]), f"{name}.json differs from result.{name}")
    original = _load_graph(artifacts / "original_adj.npz")
    modified = _load_graph(artifacts / "modified_adj.npz")
    attack, evaluation = row["attack"], row["evaluation"]
    _validate_actions(attack, original, modified, job["budget"])
    clean = _load_predictions(artifacts / "clean_predictions.npz", original.shape[0])
    adv = _load_predictions(artifacts / "combined_predictions.npz", original.shape[0])
    for key in ("node_ids", "labels"):
        _require(np.array_equal(clean[key], adv[key]), f"Prediction {key} mismatch")
    for key, field in (("node_ids", "target_nodes_hash"), ("labels", "evaluation_labels_hash")):
        _require(array_hash(clean[key]) == attack[field], f"{field} mismatch")
    _require(array_hash(clean["predictions"]) == evaluation["clean_predictions_hash"],
             "clean_predictions_hash mismatch")
    exact = prediction_metrics(clean["predictions"], adv["predictions"], adv["labels"])
    exact["combined_accuracy"] = exact["attacked_accuracy"]
    metrics = evaluation["metrics"]
    for key, value in exact.items():
        _require(key in metrics, f"Missing metric {key}")
        actual = metrics[key]
        if value is None:
            _require(actual is None, f"{key} must be undefined")
        else:
            _require(type(actual) in (int, float) and math.isfinite(actual) and
                     (type(value) is not int or type(actual) is int) and abs(value - actual) <= 1e-10,
                     f"{key} disagrees with prediction artifacts")
    # Export recomputed values, after checking the untouched saved payloads.
    evaluation["metrics"] = {**metrics, **exact}
    return row


def _per_seed_row(row):
    metrics, attack = row["evaluation"]["metrics"], row["attack"]
    return {"suite": row["suite"], "budget": row["budget"], "seed": row["seed"], "variant": row["variant"],
            "clean_accuracy": metrics["clean_accuracy"], "attacked_accuracy": metrics["combined_accuracy"],
            "drop_pp": metrics["accuracy_drop_pp"], "conditional_asr": metrics["conditional_asr"],
            "edges": attack["completed_edges"], "changed_nodes": attack["changed_feature_nodes"],
            "changed_entries": attack["changed_feature_entries"], "node_visits": attack["attribute_node_visits"],
            "structural_seconds": attack["structural_seconds"], "end_to_end_seconds": row["wall_time_seconds"],
            "process_peak_rss_mib": row["peak_rss_mib"]}


def paired_statistics(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite paired values required")
    n = len(values); mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    ci = None
    if n > 1:
        half = float(t.ppf(.975, n - 1)) * sd / math.sqrt(n)
        ci = [mean - half, mean + half]
    # Paired sign-flip randomization test; unit is a complete model/attack seed.
    if n <= 16:
        tail = sum(abs(float(np.dot(signs, values))) >= abs(float(values.sum())) - 1e-12
                   for signs in itertools.product((-1, 1), repeat=n))
        p = tail / 2**n
        method = "exact paired sign-flip"
    else:
        rng = np.random.default_rng(20260916)
        draws = rng.choice([-1, 1], size=(65536, n)) @ values
        p = (1 + int(np.sum(np.abs(draws) >= abs(values.sum()) - 1e-12))) / 65537
        method = "65536 paired sign-flips; fixed Monte Carlo seed"
    return dict(n=n, mean=mean, sd=sd, ci95=ci, p_signflip=p, p_method=method,
                paired_dz=mean/sd if sd and sd > 0 else None)


def holm(p_values):
    adjusted = [None] * len(p_values)
    running = 0.
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i: p_values[i])):
        running = max(running, min(1., (len(p_values)-rank)*p_values[index]))
        adjusted[index] = running
    return adjusted


def validate_block(rows):
    """Compare arms after individual records/artifacts have been validated."""
    errors = []
    def same(values, name):
        if len({json.dumps(v, sort_keys=True) for v in values}) != 1:
            errors.append(f"mismatched {name}")
    for row in rows:
        if row.get("status") != "complete" or row.get("protocol") != protocol_for_suite(row["suite"]):
            errors.append("incomplete or legacy protocol")
        if row.get("settings") != settings(row):
            errors.append("settings differ from protocol")
        attack = row.get("attack", {})
        if not attack.get("budget_complete") or attack.get("completed_edges") != attack.get("target_edges"):
            errors.append("edge budget incomplete")
    same([r.get("provenance") for r in rows], "source/input/environment provenance")
    for field in ("original_adj_hash", "original_features_hash", "training_nodes_hash",
                  "target_nodes_hash", "evaluation_labels_hash", "generation_model", "generation_seed", "generation_temperature"):
        values = [r.get("attack", {}).get(field) for r in rows]
        if any(v is None for v in values):
            errors.append(f"missing {field}")
        same(values, field)
    for field in ("clean_predictions_hash", "victim_seed"):
        values = [r.get("evaluation", {}).get(field) for r in rows]
        if any(v is None for v in values):
            errors.append(f"missing {field}")
        same(values, field)
    # Freeze/reproduce action trajectories only where the structural mechanism
    # is held fixed. Never impose this across the global-factorial factor.
    action_groups = {}
    for row in rows:
        key = (row["global_transition"], row["global_source"], row["search_method"])
        action_groups.setdefault(key, []).append(row)
    for group in action_groups.values():
        for field in ("actions_hash", "modified_adj_hash"):
            values = [r["attack"].get(field) for r in group]
            if any(v is None for v in values):
                errors.append(f"missing {field}")
            same(values, field)
    return sorted(set(errors))


def contrasts(suite):
    # Positive values mean the proposed alternative has LOWER attacked accuracy.
    if suite == "selectors":
        return {f"{baseline}_minus_ae_pr": {baseline:1, "ae_pr":-1}
                for baseline in ("pr", "degree", "random")}
    if suite == "ordering":
        return {"pre_minus_post": {"pre":1, "post":-1}}
    if suite == "uniform_repair":
        return {f"{method}_uniform_minus_degree": {f"{method}_uniform":1, f"{method}_degree":-1}
                for method in ("gb", "kmeans", "node_level")}
    return {
        "global_gain_with_topology_local": {"topology_ppr":1, "attribute_ppr":-1},
        "global_gain_with_attribute_local": {"topology_ae_pr":1, "attribute_ae_pr":-1},
        "local_gain_with_topology_global": {"topology_ppr":1, "topology_ae_pr":-1},
        "local_gain_with_attribute_global": {"attribute_ppr":1, "attribute_ae_pr":-1},
        "interaction_in_accuracy_drop_pp": {"attribute_ppr":1, "attribute_ae_pr":-1,
                                             "topology_ppr":-1, "topology_ae_pr":1},
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Override the per-suite default result roots")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT.parent / "followup_reports/new")
    parser.add_argument("--suites", choices=SUITES, nargs="+", default=list(SUITES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[15,16,17,18,19])
    parser.add_argument("--budgets", type=float, nargs="+")
    args = parser.parse_args(argv)
    for name in ("seeds", "suites", "budgets"):
        values = getattr(args, name)
        if values is not None and len(values) != len(set(values)):
            parser.error(f"Use distinct {name}")
    if args.budgets is not None:
        if any(not math.isfinite(b) or not 0 < b <= 1 for b in args.budgets):
            parser.error("Budgets must be finite and in (0,1]")
        if len({f"{b:g}" for b in args.budgets}) != len(args.budgets):
            parser.error("Budgets must map to distinct result directories")
    budgets_by_suite = {suite: args.budgets or ([.01,.05,.1] if suite == "uniform_repair" else [.05,.1])
                        for suite in args.suites}
    planned_contrasts = [dict(suite=suite, budget=budget, contrast=name)
                         for suite, budgets in budgets_by_suite.items() for budget in budgets
                         for name in contrasts(suite)]
    expected_blocks = sum(len(budgets) * len(args.seeds) for budgets in budgets_by_suite.values())
    protocols = {suite: protocol_for_suite(suite) for suite in args.suites}
    report_protocol = next(iter(protocols.values())) if len(set(protocols.values())) == 1 else "multiple-suite-protocols"
    report = {"protocol":report_protocol, "protocols":protocols,
              "expected_seeds": args.seeds, "expected_blocks": expected_blocks,
              "missing":[], "missing_jobs": [], "rejected_blocks":[],
              "summary":[], "contrasts":[], "complete_blocks":[], "warnings":[]}
    per_seed = []
    for suite in args.suites:
        budgets = budgets_by_suite[suite]
        for budget in budgets:
            blocks = []
            for seed in args.seeds:
                jobs = make_jobs(suite, [seed], [budget]); rows = []; export_rows = []
                for job in jobs:
                    path = (args.root or default_output(suite)) / suite / job["dataset"] / f"ptb_{budget:g}" / f"seed_{seed}" / job["variant"] / "result.json"
                    if not path.is_file():
                        report["missing"].append(str(path))
                        report["missing_jobs"].append({**job, "path": str(path)})
                        continue
                    try:
                        row = _read_validated_record(path, job)
                        exported = _per_seed_row(row)
                    except Exception as exc:
                        # Untrusted/malformed JSON, sparse, and NPZ inputs must
                        # reject this job, not prevent reporting all other jobs.
                        # Interrupts/SystemExit are deliberately not swallowed.
                        report["rejected_blocks"].append({
                            **job, "path": str(path), "error_type": type(exc).__name__,
                            "error": f"{path}: {exc}"})
                        continue
                    rows.append(row)
                    export_rows.append(exported)
                if len(rows) != len(jobs):
                    continue
                errors = validate_block(rows)
                if errors:
                    report["rejected_blocks"].append({"suite":suite,"budget":budget,"seed":seed,"error":errors}); continue
                blocks.append({r["variant"]:r for r in rows})
                report["complete_blocks"].append({"suite":suite,"budget":budget,"seed":seed})
                per_seed.extend(export_rows)
            if not blocks:
                continue
            # Different seed blocks must come from one code/data/environment cohort.
            provenance_values = {json.dumps(next(iter(b.values()))["provenance"], sort_keys=True) for b in blocks}
            if len(provenance_values) != 1:
                report["warnings"].append(f"{suite}/{budget}: mixed provenance across seeds; aggregates suppressed")
                continue
            for variant in blocks[0]:
                values = [100*b[variant]["evaluation"]["metrics"]["combined_accuracy"] for b in blocks]
                st = paired_statistics(values)
                report["summary"].append({"suite":suite,"budget":budget,"variant":variant,
                    "seeds":[b[variant]["seed"] for b in blocks], "n":st["n"], "attacked_accuracy_percent_mean":st["mean"],
                    "sd":st["sd"], "ci95":st["ci95"]})
            for name, coefficients in contrasts(suite).items():
                differences = [100*sum(weight*b[v]["evaluation"]["metrics"]["combined_accuracy"]
                                      for v,weight in coefficients.items()) for b in blocks]
                report["contrasts"].append({"suite":suite,"budget":budget,"contrast":name,
                    "seeds": [next(iter(b.values()))["seed"] for b in blocks],
                    **paired_statistics(differences)})
    # One explicit family across all requested primary accuracy contrasts/budgets.
    def contrast_key(row):
        return row["suite"], row["budget"], row["contrast"]
    estimated = {contrast_key(row): row for row in report["contrasts"]}
    adjusted = holm([estimated.get(contrast_key(row), {}).get("p_signflip", 1.0)
                     for row in planned_contrasts])
    for planned, p in zip(planned_contrasts, adjusted):
        if contrast_key(planned) in estimated:
            estimated[contrast_key(planned)]["p_holm"] = p
    report["holm_family"] = {
        "size": len(planned_contrasts), "planned_contrasts": planned_contrasts,
        "missing_contrasts": [row for row in planned_contrasts if contrast_key(row) not in estimated],
        "policy": "Unestimated planned contrasts are padded with p=1 for Holm only; they are not measured results.",
    }
    report["ready_for_full_seed_reporting"] = (
        len(report["complete_blocks"]) == expected_blocks and bool(report["summary"]) and
        not (report["missing"] or report["rejected_blocks"] or report["warnings"]))
    out = args.output_dir
    write_json(out / "summary.json", report)
    write_csv(out / "per_seed.csv", per_seed)
    write_csv(out / "accuracy_summary.csv", report["summary"])
    write_csv(out / "paired_contrasts.csv", report["contrasts"])
    lines = ["# AE-PR 后续实验汇总", "", f"完整配对区组：{len(report['complete_blocks'])}；缺失运行：{len(report['missing'])}；拒绝记录/区组：{len(report['rejected_blocks'])}。", "",
        "只汇总完整且配置、输入、目标节点、clean 预测及应保持一致的结构轨迹均匹配的区组。",
        "单位是独立模型/攻击 seed，不把节点当作独立重复。均值/样本标准差；95% CI 使用 t 分布。配对 sign-flip 检验依赖零假设下符号可交换性；Holm 家族覆盖本次请求的所有主准确率对比和预算。少量 seed 的 CI 与检验仅作探索性证据。",
        "未能估计的计划对比仅在 Holm 校正中以 p=1 补齐，不缩小计划检验家族；这些占位值不是实验结果。",
        "各方法共享预算上限和可行性规则；实际修改节点/条目/访问次数在 per_seed.csv 中披露，不声称实际消耗相等。",
        "structural_seconds 包含内部训练、全局评分、建树与端点搜索；总时间另列，进程峰值内存不包括独立 LLM 服务。", "",
        "| 套件 | 预算 | 变体 | n | attacked accuracy (%) mean ± SD |", "|---|---:|---|---:|---:|"]
    for r in report["summary"]:
        sd = f"{r['sd']:.4f}" if r["sd"] is not None else "未定义"
        lines.append(f"| {r['suite']} | {r['budget']:g} | {r['variant']} | {r['n']} | {r['attacked_accuracy_percent_mean']:.4f} ± {sd} |")
    lines += ["", "未执行完整实验时，本报告不会产生替代论文数字。完整缺失清单、拒绝原因及配对差值见 summary.json。"]
    (out / "summary.md").write_text("\n".join(lines)+"\n")
    print(f"Wrote {out}; complete blocks={len(report['complete_blocks'])}; missing={len(report['missing'])}")
    return 0 if report["ready_for_full_seed_reporting"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
