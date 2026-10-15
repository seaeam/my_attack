"""Validation-only development rankings for an explicitly planned Post sweep."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.action_state_validation import INPUT_HASHES
from gbhaa_experiments.followup_protocol import array_hash
from gbhaa_experiments.post_tuning_protocol import PROTOCOL, GUIDED_SELECTOR_RULES, COVERAGE_SELECTOR_RULES, EDIT_PROFILES, TRANSFER_PROFILES, OBJECTIVE_PROFILES, reference_profile, needs_surrogate
from gbhaa_experiments.post_tuning_validation import validate_job, validate_result
from gbhaa_experiments.runner import write_json
from gbhaa_experiments.summarize_followup import _canonical, _read_json, _require, write_csv

DEFAULT_ROOT = REPO_ROOT / "gbhaa_experiments/post_tuning_results"
DEFAULT_OUTPUT = REPO_ROOT / "gbhaa_experiments/followup_reports/post_tuning"
CRITERION = "mean validation accuracy ascending; mean validation NLL descending; stable profile_id ascending"


def result_path(root, job):
    return (Path(root) / job["dataset"] / f"source_ptb_{job['source_budget']:g}" /
            f"steps_{job['max_actions']}" / f"seed_{job['seed']}" / job["trial_id"] / "result.json")


def _cell(job):
    return job["dataset"], job["source_budget"], job["max_actions"]


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                sd=float(values.std(ddof=1)) if len(values) > 1 else None)


def _clean_identity(path):
    with np.load(path.parent / "artifacts/clean_validation_predictions.npz", allow_pickle=False) as data:
        return {key: array_hash(data[key]) for key in ("node_ids", "labels", "predictions", "log_probabilities")}


def _block_errors(rows, paths):
    errors = []

    def same(values, name):
        if len({_canonical(value) for value in values}) != 1:
            errors.append(f"mismatched {name}")

    same([row["provenance"] for row in rows], "source/input/environment provenance")
    same([row["parent_fingerprint"] for row in rows], "parent fingerprint")
    for key in (*INPUT_HASHES, "actions", "actions_hash", "modified_adj_hash", "effective_budget",
                "validation_nodes_hash", "holdout_nodes_hash", "prefix_length", "generation_seed"):
        same([row["attack"][key] for row in rows], key)
    for key in ("clean_predictions_hash", "victim_seed"):
        same([row["evaluation"][key] for row in rows], key)
    same([_clean_identity(path) for path in paths], "clean validation outputs")
    guided = [row for row in rows if needs_surrogate(row["config"])]
    if guided:
        same([row["attack"].get("selector_surrogate") for row in guided], "fixed selector surrogate/pseudo-labels/targets")
    editing = [row for row in rows if "editor_gain" in row["config"]]
    if editing:
        snapshots=[row['attack'].get('editor_snapshot',{}) for row in editing]
        for key in ('vectorizer_sha256','vocabulary'):
            same([s.get(key) for s in snapshots],f'editor {key}')
        for policy in {row['config'].get('editor_proxy','source') for row in editing}:
            same([row['attack'].get('editor_snapshot') for row in editing
                  if row['config'].get('editor_proxy','source')==policy],f'fixed editor model/vectorizer ({policy})')
    return errors


def _per_trial(job, status, path, row=None, error=None):
    exported = {key: job[key] for key in ("dataset", "source_budget", "max_actions", "seed", "trial_id", "arm", "profile_id", "config")}
    exported.update(status=status, error=error, result_path=str(path))
    fields = ("effective_budget", "clean_validation_accuracy_percent", "validation_accuracy_percent", "clean_validation_nll",
              "validation_nll", "accuracy_drop_pp", "completed_edges", "changed_feature_nodes", "changed_feature_entries",
              "feature_l1", "feature_l2", "feature_linf", "attribute_node_visits", "visited_attribute_nodes", "successful_writes",
              "mean_unique_coverage_nodes", "mean_unique_coverage_fraction", "mean_endpoint_occupancy_fraction",
              "probe_changed_set_fraction", "mean_probe_jaccard", "mean_mechanism_fallback_fraction",
              "mean_mechanism_positive_nonrestart_nodes",
              "fresh_node_write_fraction", "repeated_node_write_fraction", "mean_fresh_pool_nodes",
              "mean_selected_fresh_nodes", "mean_fresh_quota_shortfall", "fresh_quota_shortfall_rounds",
              "structural_seconds", "wall_time_seconds", "peak_rss_mib")
    exported.update(dict.fromkeys(fields))
    if row is None:
        return exported
    attack, metrics = row["attack"], row["evaluation"]["metrics"]
    exported.update(clean_validation_accuracy_percent=100*metrics["clean_validation_accuracy"],
                    validation_accuracy_percent=100*metrics["validation_accuracy"],
                    clean_validation_nll=metrics["clean_validation_nll"], validation_nll=metrics["validation_nll"],
                    accuracy_drop_pp=metrics["accuracy_drop_pp"], successful_writes=attack["attribute_counts"]["feature_writes"],
                    wall_time_seconds=row["wall_time_seconds"], peak_rss_mib=row["peak_rss_mib"])
    for key in ("effective_budget", "completed_edges", "changed_feature_nodes", "changed_feature_entries", "feature_l1",
                "feature_l2", "feature_linf", "attribute_node_visits", "visited_attribute_nodes", "structural_seconds"):
        exported[key] = attack[key]
    visits = attack["attribute_node_visits"]
    if visits:
        exported["fresh_node_write_fraction"] = attack["visited_attribute_nodes"] / visits
        exported["repeated_node_write_fraction"] = 1 - exported["fresh_node_write_fraction"]
    rounds = _read_json(path.parent / "artifacts/rounds.json")["rounds"]
    seen, coverage, occupancy = set(), [], []
    for diagnostic in rounds:
        selected = set(diagnostic["selected_nodes"])
        seen.update(selected)
        coverage.append(len(seen))
        occupancy.append(len(selected & set(diagnostic["action"][:2])) / len(selected))
    exported["mean_unique_coverage_nodes"] = float(np.mean(coverage))
    exported["mean_unique_coverage_fraction"] = float(np.mean(coverage)) / rounds[0]["candidate_universe"]
    exported["mean_endpoint_occupancy_fraction"] = float(np.mean(occupancy))
    exported["probe_changed_set_fraction"] = float(np.mean([
        set(d["probe_pre_nodes"]) != set(d["probe_post_nodes"]) for d in rounds]))
    exported["mean_probe_jaccard"] = float(np.mean([d["probe_jaccard"] for d in rounds]))
    if "selector_rule" in job["config"]:
        key = "parallel" if job["arm"] == "parallel" else ("post" if job["arm"] == "endpoint_post" else "pre")
        selections = [d["mechanism_traces"][key]["selection"] for d in rounds]
        exported["mean_mechanism_fallback_fraction"] = float(np.mean([s["fallback_count"]/4 for s in selections]))
        exported["mean_mechanism_positive_nonrestart_nodes"] = float(np.mean([s["positive_nonrestart_count"] for s in selections]))
        if job["config"]["selector_rule"] in COVERAGE_SELECTOR_RULES:
            exported["mean_fresh_pool_nodes"] = float(np.mean([s["fresh_pool_count"] for s in selections]))
            exported["mean_selected_fresh_nodes"] = float(np.mean([s["selected_fresh_count"] for s in selections]))
            if job["config"]["selector_rule"] == "frontier_editgain_fresh3":
                shortfalls = [s["fresh_quota_shortfall"] for s in selections]
                exported["mean_fresh_quota_shortfall"] = float(np.mean(shortfalls))
                exported["fresh_quota_shortfall_rounds"] = sum(v > 0 for v in shortfalls)
    if "editor_gain" in job["config"]:
        traces = [d["editing"] for d in rounds]
        node_traces = [n for trace in traces for n in trace["nodes"]]
        exported.update(editor_mean_distinct_responses=float(np.mean([v["distinct_responses"] for v in traces])),
            editor_mean_distinct_candidates=float(np.mean([n["distinct_candidates"] for n in node_traces])),
            editor_actual_gain_sum=sum(v["actual_gain"] for v in traces),
            editor_negative_gain_rounds=sum(v["actual_gain"] < 0 for v in traces),
            editor_projection_fraction_mean=float(np.mean([n["projections"][n["chosen"]]["projection_fraction"] for n in node_traces])),
            editor_actual_planned_additions=sum(n["projections"][n["chosen"]]["actual_added"] for n in node_traces),
            editor_actual_planned_removals=sum(n["projections"][n["chosen"]]["actual_dropped"] for n in node_traces),
            editor_unplanned_decreased_entries=sum(n["projections"][n["chosen"]]["unplanned_decreased_entries"] for n in node_traces),
            editor_unplanned_increased_entries=sum(n["projections"][n["chosen"]]["unplanned_increased_entries"] for n in node_traces),
            editor_logical_requests=3*len(traces),
            editor_changed_writes=attack['attribute_counts']['feature_changes'])
        with np.load(path.parent/'artifacts/features.npz',allow_pickle=False) as z:
            current=z['original'].copy();final=z['modified']
        width=min(current.shape[1],len(attack['editor_snapshot']['vocabulary']))
        exported.update(editor_proxy_policy=job['config'].get('editor_proxy','source'),
            editor_edit_policy='planned_residual' if job['config'].get('editor_residual') else 'full_replacement',
            editor_unmapped_final_l1=float(np.abs(final[:,width:]-current[:,width:]).sum()),
            editor_unmapped_final_decreases=int(np.sum(final[:,width:]<current[:,width:])),
            editor_chosen_slot_counts=[sum(n['chosen']==i for n in node_traces) for i in range(3)])
        decreases=0;total_l1=0.
        with np.load(path.parent/'artifacts/feature_updates.npz',allow_pickle=False) as z:
            for nodes,values in zip(z['nodes'],z['values']):
                delta=values-current[nodes]
                decreases+=int(np.sum(delta[:,width:]<0));total_l1+=float(np.abs(delta).sum())
                current[nodes]=values
        exported.update(editor_unmapped_decreases_sum=decreases,editor_actual_l1_sum=total_l1)
        if 'editor_objective' in job['config']:
            objectives=[v['objective'] for v in traces]
            exported.update(editor_objective=job['config']['editor_objective'],
                editor_local_mix=job['config']['editor_local_mix'],
                editor_mean_effective_targets=float(np.mean([v['effective_target_count'] for v in objectives])),
                editor_uniform_fallback_rounds=sum(v['uniform_fallback'] for v in objectives))
    return exported


def _compare(candidate, controls, profile_id, control_profile_id, control_arm, mode, cell,
             expected_seeds=None):
    seeds = sorted(set(candidate) & set(controls))
    expected_seeds = sorted(set(candidate) | set(controls)) if expected_seeds is None else sorted(expected_seeds)
    accuracy, nll, wins, losses, ties, differences = [], [], 0, 0, 0, []
    for seed in seeds:
        post = candidate[seed]["evaluation"]["metrics"]
        control = controls[seed]["evaluation"]["metrics"]
        acc_diff = 100*(control["validation_accuracy"] - post["validation_accuracy"])
        nll_diff = post["validation_nll"] - control["validation_nll"]
        accuracy.append(acc_diff); nll.append(nll_diff)
        accuracy_outcome = (acc_diff > 0) - (acc_diff < 0)
        outcome = accuracy_outcome
        if outcome == 0:
            outcome = (nll_diff > 0) - (nll_diff < 0)
        wins += outcome > 0; losses += outcome < 0; ties += outcome == 0
        differences.append(dict(seed=seed, control_minus_post_accuracy_pp=acc_diff,
                                post_minus_control_nll=nll_diff,
                                accuracy_outcome="win" if accuracy_outcome > 0 else "loss" if accuracy_outcome < 0 else "tie",
                                outcome="win" if outcome > 0 else "loss" if outcome < 0 else "tie"))
    acc_stats, nll_stats = _stats(accuracy), _stats(nll)
    ci95 = None
    if len(seeds) > 1:
        half = float(t.ppf(.975, len(seeds)-1)) * acc_stats['sd'] / np.sqrt(len(seeds))
        ci95 = [acc_stats['mean']-half, acc_stats['mean']+half]
    return dict(dataset=cell[0], source_budget=cell[1], max_actions=cell[2], profile_id=profile_id,
                control_profile_id=control_profile_id, control_arm=control_arm, comparison_mode=mode,
                expected_seeds=expected_seeds, expected_n=len(expected_seeds),
                status="complete" if seeds == expected_seeds else "partial",
                seeds=seeds, n=len(seeds), valid_n=len(seeds), wins=int(wins), losses=int(losses), ties=int(ties),
                accuracy_wins=sum(v > 0 for v in accuracy), accuracy_losses=sum(v < 0 for v in accuracy),
                accuracy_ties=sum(v == 0 for v in accuracy),
                control_minus_post_accuracy_pp_mean=acc_stats["mean"], control_minus_post_accuracy_pp_sd=acc_stats["sd"],
                control_minus_post_accuracy_pp_ci95=ci95,
                ci_method="unadjusted paired Student-t across seeds; descriptive development interval",
                post_minus_control_nll_mean=nll_stats["mean"], post_minus_control_nll_sd=nll_stats["sd"], paired_differences=differences)


def summarize(root):
    """Use plan.json as the complete requested matrix; never infer absent trials."""
    root = Path(root)
    report = dict(protocol=PROTOCOL, selection_split="validation", criterion=CRITERION,
                  status="incomplete", expected_jobs=0, per_trial=[], ranking=[], comparisons=[],
                  missing_jobs=[], rejected_jobs=[], rejected_blocks=[], warnings=[], recommendations=[])
    try:
        plan = _read_json(root / "plan.json")
        _require(plan.get("protocol") == PROTOCOL, "plan protocol mismatch")
        jobs = plan.get("jobs")
        _require(isinstance(jobs, list) and bool(jobs), "plan jobs must be a nonempty list")
        for job in jobs:
            validate_job(job)
        reference_profile([j["profile_id"] for j in jobs])
        _require(len({_canonical(job) for job in jobs}) == len(jobs), "duplicate job in plan")
        _require(len({str(result_path(root, job)) for job in jobs}) == len(jobs), "duplicate trial path in plan")
        _require(isinstance(plan.get("provenance"), dict) and bool(plan["provenance"]), "plan provenance missing")
    except Exception as exc:
        report["rejected_blocks"].append(dict(error=f"plan.json: {exc}"))
        return report
    report["expected_jobs"] = len(jobs)
    valid, paths, expected_blocks, groups = {}, {}, defaultdict(list), defaultdict(list)
    for job in jobs:
        key = (*_cell(job), job["seed"])
        expected_blocks[key].append(job)
        groups[_cell(job)].append(job)
        path = result_path(root, job)
        if not path.is_file():
            report["missing_jobs"].append({**job, "path":str(path)})
            report["per_trial"].append(_per_trial(job, "missing", path))
            continue
        try:
            row = validate_result(path, job)
            valid[str(path)], paths[str(path)] = row, path
            report["per_trial"].append(_per_trial(job, "complete", path, row))
        except Exception as exc:
            report["rejected_jobs"].append({**job, "path":str(path), "error":str(exc), "error_type":type(exc).__name__})
            report["per_trial"].append(_per_trial(job, "rejected", path, error=str(exc)))
    for key, block_jobs in expected_blocks.items():
        block_paths = [result_path(root, job) for job in block_jobs]
        block_rows = [valid[str(path)] for path in block_paths if str(path) in valid]
        errors = []
        anchor = reference_profile([j["profile_id"] for j in block_jobs])
        if not any(j["profile_id"] == anchor and j["arm"] == "endpoint_post" for j in block_jobs):
            errors.append(f"{anchor} Post missing from plan")
        if block_rows:
            # Missing trials must not hide a mismatch among available results.
            errors.extend(_block_errors(block_rows, [path for path in block_paths if str(path) in valid]))
        if errors:
            report["rejected_blocks"].append(dict(dataset=key[0], source_budget=key[1], max_actions=key[2], seed=key[3], errors=errors))
    for cell, cell_jobs in sorted(groups.items()):
        seeds = sorted({job["seed"] for job in cell_jobs})
        planned_sets = [{(j["profile_id"], j["arm"], _canonical(j["config"])) for j in cell_jobs if j["seed"] == seed} for seed in seeds]
        if any(values != planned_sets[0] for values in planned_sets):
            report["rejected_blocks"].append(dict(dataset=cell[0], source_budget=cell[1], max_actions=cell[2], error="trial/config matrix differs across seeds"))
        cell_rows = [valid[str(result_path(root, job))] for job in cell_jobs if str(result_path(root, job)) in valid]
        if any(_canonical(row["provenance"]) != _canonical(plan["provenance"]) for row in cell_rows):
            report["rejected_blocks"].append(dict(dataset=cell[0], source_budget=cell[1], max_actions=cell[2], error="trial provenance differs from plan/across seeds"))
        by_trial = defaultdict(dict)
        for row in cell_rows:
            by_trial[row["profile_id"], row["arm"]][row["seed"]] = row
        anchor = reference_profile([j["profile_id"] for j in cell_jobs])
        ranking = []
        profiles = sorted({j["profile_id"] for j in cell_jobs if j["arm"] == "endpoint_post"})
        for profile_id in profiles:
            rows = by_trial[profile_id, "endpoint_post"]
            config = next(j["config"] for j in cell_jobs if j["profile_id"] == profile_id)
            acc = _stats([100*r["evaluation"]["metrics"]["validation_accuracy"] for r in rows.values()])
            nll = _stats([r["evaluation"]["metrics"]["validation_nll"] for r in rows.values()])
            ranking.append(dict(dataset=cell[0], source_budget=cell[1], max_actions=cell[2], rank=None,
                profile_id=profile_id, config=config, n=acc["n"], expected_n=len(seeds), seeds=sorted(rows),
                status="complete" if len(rows) == len(seeds) else "partial", validation_accuracy_percent_mean=acc["mean"],
                validation_accuracy_percent_sd=acc["sd"], validation_nll_mean=nll["mean"], validation_nll_sd=nll["sd"]))
            if profile_id != anchor:
                report["comparisons"].append(_compare(rows, by_trial[anchor, "endpoint_post"], profile_id, anchor, "endpoint_post", "Post profile versus reference Post", cell, seeds))
            if profile_id in TRANSFER_PROFILES[1:] and any(j['profile_id']=='edit_both' for j in cell_jobs):
                report['comparisons'].append(_compare(rows,by_trial['edit_both','endpoint_post'],profile_id,
                    'edit_both','endpoint_post','Post transfer policy versus original combined editor',cell,seeds))
            if profile_id in COVERAGE_SELECTOR_RULES and any(
                    j["profile_id"] == "frontier_editgain" and j["arm"] == "endpoint_post" for j in cell_jobs):
                report["comparisons"].append(_compare(rows, by_trial["frontier_editgain", "endpoint_post"],
                    profile_id, "frontier_editgain", "endpoint_post",
                    "Post coverage profile versus original editgain Post", cell, seeds))
            for arm in ("endpoint_pre", "parallel"):
                matched = any(j["profile_id"] == profile_id and j["arm"] == arm for j in cell_jobs)
                control_profile = profile_id if matched else anchor
                if any(j["profile_id"] == control_profile and j["arm"] == arm for j in cell_jobs):
                    report["comparisons"].append(_compare(rows, by_trial[control_profile, arm], profile_id, control_profile, arm,
                        "matched configuration" if matched else "asymmetric exploratory comparison against baseline control", cell, seeds))
        ranking.sort(key=lambda r: (float("inf") if r["validation_accuracy_percent_mean"] is None else r["validation_accuracy_percent_mean"],
                                    float("inf") if r["validation_nll_mean"] is None else -r["validation_nll_mean"], r["profile_id"]))
        for index, row in enumerate(ranking, 1):
            if row["n"]:
                row["rank"] = index
        report["ranking"].extend(ranking)
    complete = len(valid) == len(jobs) and not (report["missing_jobs"] or report["rejected_jobs"] or report["rejected_blocks"])
    report["status"] = "complete" if complete else "incomplete"
    for comparison in report['comparisons']:
        comparison['matrix_status'] = report['status']
        if any(all(block.get(key, comparison[key]) == comparison[key]
                   for key in ('dataset', 'source_budget', 'max_actions'))
               for block in report['rejected_blocks']):
            comparison['status'] = 'rejected'
            comparison['valid_n'] = 0
            comparison['control_minus_post_accuracy_pp_ci95'] = None
    report["factorial_effects"] = [];report['transfer_effects']=[]
    report['objective_effects']=[];report['joint_advantage']=[]
    for cell, cell_jobs in groups.items():
        planned={j['profile_id'] for j in cell_jobs}
        transfer=bool(planned.intersection(TRANSFER_PROFILES[1:]))
        objective=bool(planned.intersection(OBJECTIVE_PROFILES[1:]))
        if not planned.intersection(EDIT_PROFILES) and not transfer and not objective:
            continue
        from gbhaa_experiments.post_editing_summary import factorial_effects,transfer_effects,objective_effects,joint_advantage
        by_trial = defaultdict(dict)
        for job in cell_jobs:
            row = valid.get(str(result_path(root, job)))
            if row is not None:
                by_trial[row["profile_id"], row["arm"]][row["seed"]] = row
        seeds=sorted({j['seed'] for j in cell_jobs})
        if (not transfer and not objective) or set(EDIT_PROFILES).issubset(planned):
            report["factorial_effects"].extend(factorial_effects(by_trial,seeds,cell,complete))
        if transfer and (not objective or set(TRANSFER_PROFILES).issubset(planned)):
            report['transfer_effects'].extend(transfer_effects(by_trial,seeds,cell,complete))
        if objective:
            report['objective_effects'].extend(objective_effects(by_trial,seeds,cell,complete))
            report['joint_advantage'].extend(joint_advantage(by_trial,seeds,cell,complete,sorted(planned)))
    if complete:
        for cell in sorted(groups):
            winner = next(row for row in report["ranking"] if _cell(row) == cell)
            report["recommendations"].append({**winner, "selection_split":"validation", "development_only":True})
    report["warnings"] = ["Development selection uses validation data only. Seeds 15–19 already have historical evaluations and are not a pristine holdout.",
        "A short prefix is not the full source-budget attack. One-seed results do not establish stable performance; sample SD is undefined at n=1.",
        "Post-only profiles versus baseline Pre/Parallel are exploratory. Matched configurations are required to attribute differences to graph timing.",
        "Paired 95% t intervals use seed-level ACC differences and are unadjusted, descriptive development intervals. They do not correct for searching many profiles or establish independent test-set superiority.",
        "Partial rankings retain valid observations, but any missing, rejected or unmatched requested trial suppresses every recommendation."]
    return report


def _markdown(report):
    lines = ["# Post tuning development sweep", "", f"Status: **{report['status']}**. Expected trials: {report['expected_jobs']}; missing: {len(report['missing_jobs'])}; rejected: {len(report['rejected_jobs'])}.",
        "", f"Selection: {CRITERION}. ACC columns are percentages. NLL is mean negative log probability of the validation label (higher means a stronger attack).", "",
        "| Dataset | Source budget | Actions | Profile | Valid/expected seeds | Validation ACC % ± sample SD | Validation NLL | Status |",
        "|---|---:|---:|---|---:|---:|---:|---|"]
    for row in report["ranking"]:
        mean = "unavailable" if row["validation_accuracy_percent_mean"] is None else f"{row['validation_accuracy_percent_mean']:.4f}"
        sd = "undefined" if row["validation_accuracy_percent_sd"] is None else f"{row['validation_accuracy_percent_sd']:.4f}"
        nll = "unavailable" if row["validation_nll_mean"] is None else f"{row['validation_nll_mean']:.6f}"
        lines.append(f"| {row['dataset']} | {row['source_budget']:g} | {row['max_actions']} | {row['profile_id']} | {row['n']}/{row['expected_n']} | {mean} ± {sd} | {nll} | {row['status']} |")
    lines += ["", "## Paired ACC comparisons", "",
        "Positive control-minus-Post ACC (percentage points) favors Post. ACC-only W/L/T does not count an NLL-only improvement as an ACC win. Intervals are unadjusted seed-level development intervals; n=1 is undefined.", "",
        "| Dataset / actions | Post profile | Control profile / arm | Mode | Valid/observed/expected seeds | ACC difference mean ± SD (pp) | 95% CI (pp) | ACC-only W/L/T | Status |",
        "|---|---|---|---|---:|---:|---|---|---|"]
    for row in report['comparisons']:
        mean = 'unavailable' if row['control_minus_post_accuracy_pp_mean'] is None else f"{row['control_minus_post_accuracy_pp_mean']:.4f}"
        sd = 'undefined' if row['control_minus_post_accuracy_pp_sd'] is None else f"{row['control_minus_post_accuracy_pp_sd']:.4f}"
        ci = row['control_minus_post_accuracy_pp_ci95']
        interval = 'undefined' if ci is None else f'[{ci[0]:.4f}, {ci[1]:.4f}]'
        lines.append(f"| {row['dataset']} / {row['max_actions']} | {row['profile_id']} | {row['control_profile_id']} / {row['control_arm']} | {row['comparison_mode']} | {row['valid_n']}/{row['n']}/{row['expected_n']} | {mean} ± {sd} | {interval} | {row['accuracy_wins']}/{row['accuracy_losses']}/{row['accuracy_ties']} | {row['status']} |")
    lines += ["", "paired_comparisons.csv and summary.json retain paired seed differences, ACC-only outcomes, sample SD and 95% intervals. Historical wins/losses/ties and outcome fields still use ACC first, then NLL for tied ACC; accuracy_wins/losses/ties and accuracy_outcome use ACC only. n counts observed pairs; valid_n is zero for rejected cells and their intervals are suppressed. Their remaining deltas and counts are audit data only. Partial or rejected comparisons cannot establish an advantage.",
        "", "per_trial.csv records source and effective budgets, changed feature nodes/entries/L1/L2/Linf, successful writes, runtime, and coverage. Mean unique coverage averages cumulative distinct selected nodes over rounds; endpoint occupancy is the fraction of each round's four selections at its two action endpoints.",
        "", "Fresh/repeated write fractions count first/repeat visits across the full trajectory. Coverage profiles additionally report fresh nodes in the unchanged diffusion pool and selected fresh nodes per round. A fresh3 quota shortfall means fewer than three fresh nodes in that pool before selection; it is not the final write-count deficit. When the original editgain profile is planned, new coverage Post profiles are also compared with its Post arm.",
        "", "## Interpretation", "", *[f"- {warning}" for warning in report["warnings"]]]
    if report.get("factorial_effects"):
        lines += ["", "## Editor factorial (ACC percentage points)", "",
            "c=edit_control, g=edit_gain, w=edit_words, b=edit_both. Gain=((c-g)+(w-b))/2; words=((c-w)+(g-b))/2; interaction=g+w-c-b. Positive favors the editing change. All contrasts are formed within seed and arm first. Incomplete matrices suppress effect estimates.", "",
            "| Arm | Effect | Valid/expected seeds | Mean ± SD (pp) | 95% CI (pp) | Status |",
            "|---|---|---:|---|---|---|"]
        for row in report["factorial_effects"]:
            lines.append(f"| {row['arm']} | {row['effect']} | {row['valid_n']}/{row['expected_n']} | {row['mean_pp']} ± {row['sd_pp']} | {row['ci95_pp']} | {row['status']} |")
        lines += ["", "New editor cells share three real requests/round (50 output tokens each), with edit_control as their same-budget reference. They do not reuse the historical one-request baseline. Editor gradients and candidate losses use the arm's diffusion graph; this evaluates the full local editor's graph state, not diffusion timing alone. Candidate duplicates, projection fractions and actual signed gains are retained in per_trial.csv and rounds.json."]
    if report.get('transfer_effects'):
        lines += ['', '## Transfer policy factorial (ACC percentage points)', '',
            'Anchor c=edit_both; r=transfer_relu; s=transfer_residual; b=transfer_both. Proxy=((c-r)+(s-b))/2; residual=((c-s)+(r-b))/2; interaction=r+s-c-b. Positive means lower attacked ACC. edit_control is the separate prior best reference, not a factorial cell. The proxy factor bundles ReLU architecture, fitting and independent initialization; it does not isolate activation alone.', '',
            '| Dataset / actions | Arm | Effect | Valid/expected seeds | Mean ± SD (pp) | 95% CI | Status |',
            '|---|---|---|---:|---|---|---|']
        for row in report['transfer_effects']:
            lines.append(f"| {row['dataset']} / {row['max_actions']} | {row['arm']} | {row['effect']} | {row['valid_n']}/{row['expected_n']} | {row['mean_pp']} ± {row['sd_pp']} | {row['ci95_pp']} | {row['status']} |")
        lines += ['', 'All policies use three real requests and four writes per action. Residual editing is a planned feature patch through the existing partial vocabulary, not full text replacement or verified semantic remapping. Unplanned coordinates remain unchanged. Gradients and candidate scores follow the arm graph. Actual selections and edit amplitudes may diverge with evolving features; per_trial.csv retains coverage, request counts, support changes and L1. The separate proxy is fitted on clean data with training labels only; it does not model subsequent victim retraining.']
    if report.get('objective_effects'):
        lines += ['', '## ReLU objective factorial (ACC percentage points)', '',
            'Cells: transfer_relu, objective_margin, objective_local, objective_both. Margin and local main effects average their two paired simple effects; interaction = margin + local - baseline - both. Positive favors the factor. Every cell uses the same ReLU proxy and full replacement policy.', '',
            '| Dataset / actions | Arm | Effect | Valid/expected seeds | Mean ± SD (pp) | 95% CI | Status |',
            '|---|---|---|---:|---|---|---|']
        for row in report['objective_effects']:
            lines.append(f"| {row['dataset']} / {row['max_actions']} | {row['arm']} | {row['effect']} | {row['valid_n']}/{row['expected_n']} | {row['mean_pp']} ± {row['sd_pp']} | {row['ci95_pp']} | {row['status']} |")
        lines += ['', 'Capped margin = -relu(logp[pseudo] - max_other_logp). Local weights mix half uniform and half normalized arm-local diffusion mass on the unchanged nontraining targets, frozen at round start. Word gradients and candidate scores use the same objective and graph. Losses/gains from different objectives cannot be directly compared. This evaluates the full local editor state, not diffusion timing alone.', '',
            '## Post advantage against both matched arms', '',
            'Per seed minimum = min(Pre ACC - Post ACC, Parallel ACC - Post ACC). A strict joint win requires both differences > 0; ties do not count. These are development checks, not proof of stability. Missing or rejected planned trials suppress aggregates.', '',
            '| Dataset / actions | Profile | Valid/expected seeds | Strict joint wins | Minimum advantage mean ± SD (pp) | 95% CI | Worst seed minimum (pp) | Status |',
            '|---|---|---:|---:|---|---|---|---|']
        for row in report['joint_advantage']:
            lines.append(f"| {row['dataset']} / {row['max_actions']} | {row['profile_id']} | {row['valid_n']}/{row['expected_n']} | {row['strict_joint_wins']} | {row['minimum_advantage_mean_pp']} ± {row['minimum_advantage_sd_pp']} | {row['minimum_advantage_ci95_pp']} | {row['worst_seed_minimum_advantage_pp']} | {row['status']} |")
    if report["rejected_blocks"]:
        lines += ["", "## Rejected plan or matched blocks", "", *[f"- {json.dumps(value, ensure_ascii=False)}" for value in report["rejected_blocks"]]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = summarize(args.root)
    recommendation = {key: report[key] for key in ("protocol", "status", "selection_split", "criterion", "recommendations")}
    write_json(args.output_dir / "summary.json", report)
    write_json(args.output_dir / "recommendation.json", recommendation)
    # Optional objective diagnostics and incomplete trials have different keys.
    # CSV needs the union even when the first planned row is a missing baseline.
    columns=dict.fromkeys(key for row in report['per_trial'] for key in row)
    write_csv(args.output_dir / "per_trial.csv", [{**columns,**row} for row in report['per_trial']])
    write_csv(args.output_dir / "ranking.csv", report["ranking"])
    write_csv(args.output_dir / "paired_comparisons.csv", report["comparisons"])
    if report.get("factorial_effects"):
        write_csv(args.output_dir / "factorial_effects.csv", report["factorial_effects"])
    if report.get('transfer_effects'):
        write_csv(args.output_dir/'transfer_effects.csv',report['transfer_effects'])
    for key in ('objective_effects','joint_advantage'):
        if report.get(key):write_csv(args.output_dir/f'{key}.csv',report[key])
    (args.output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_dir}; status={report['status']}; trials={report['expected_jobs']}")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
