"""Complete paired-seed summaries for Parallel, Endpoint-Pre and Endpoint-Post."""
from __future__ import annotations

import argparse
from pathlib import Path

from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.action_state_protocol import ARMS, DATASETS, PROTOCOL, make_jobs
from gbhaa_experiments.action_state_validation import INPUT_HASHES, validate_result
from gbhaa_experiments.runner import write_json
from gbhaa_experiments.summarize_followup import _canonical, holm, paired_statistics, write_csv

DEFAULT_ROOT = REPO_ROOT / "gbhaa_experiments/followup_results_action_state"
CONTRASTS = {
    "pre_minus_post": ("endpoint_pre", "endpoint_post", "primary"),
    "parallel_minus_post": ("parallel", "endpoint_post", "secondary"),
    "parallel_minus_pre": ("parallel", "endpoint_pre", "secondary"),
}


def validate_block(rows):
    """Validate shared identities after checking each stage's actual artifacts."""
    errors = []
    by_variant = {row["variant"]: row for row in rows}
    if len(rows) != 4 or set(by_variant) != {"structure", *ARMS}:
        return ["missing or duplicate stage in complete block"]

    def same(values, name):
        if len({_canonical(value) for value in values}) != 1:
            errors.append(f"mismatched {name}")

    same([row["provenance"] for row in rows], "source/input/environment provenance")
    for key in ("dataset", "seed", "budget", "settings"):
        same([row[key] for row in rows], key)
    stages = [row["trajectory"] if row["variant"] == "structure" else row["attack"] for row in rows]
    for key in (*INPUT_HASHES, "actions_hash", "modified_adj_hash", "actions", "target_edges"):
        same([stage[key] for stage in stages], key)
    arms = [by_variant[variant] for variant in ARMS]
    for key in ("clean_predictions_hash", "victim_seed"):
        same([row["evaluation"][key] for row in arms], key)
    for key in ("generation_seed", "generation_model", "generation_temperature"):
        same([row["attack"][key] for row in arms], key)
    return errors


def _per_seed(row):
    attack, metrics = row["attack"], row["evaluation"]["metrics"]
    exported = {key: row[key] for key in ("dataset", "budget", "seed", "variant")}
    exported.update(clean_accuracy_percent=100*metrics["clean_accuracy"],
                    attacked_accuracy_percent=100*metrics["combined_accuracy"],
                    drop_pp=metrics["accuracy_drop_pp"], conditional_asr=metrics["conditional_asr"])
    for field in ("completed_edges", "changed_feature_nodes", "changed_feature_entries", "feature_l1",
                  "feature_l2", "feature_linf", "attribute_node_visits", "visited_attribute_nodes", "structural_seconds"):
        exported[field] = attack[field]
    exported.update(end_to_end_seconds=row["wall_time_seconds"], process_peak_rss_mib=row["peak_rss_mib"])
    return exported


def _markdown(report):
    lines = ["# Matched action-state experiment", "",
        f"Complete three-arm seed blocks: {len(report['complete_blocks'])}; missing stages: {len(report['missing_jobs'])}; rejected records/blocks: {len(report['rejected_blocks'])}.", "",
        "The sampling unit is a complete model/attack seed. Datasets and budgets are summarized separately.",
        "ACC is recomputed from target-level predictions. Means, sample SDs and t-based 95% mean CIs are reported in percent; paired differences are percentage points.",
        "n < 5 is preliminary; SD and CI are undefined at n = 1. The five-seed t interval is approximate.",
        "Pre minus Post is primary; Parallel minus Post and Parallel minus Pre are descriptive secondary contrasts. Positive means the second arm has lower attacked ACC.",
        "The primary paired sign-flip test assumes sign exchangeability under the null. Holm covers all requested dataset/budget cells; missing cells use p = 1 only for correction and are not measurements.",
        "The comparison estimates cumulative localization-policy effects under a fixed structural plan. Intermediate features can diverge between arms; this does not prove that post-action diffusion is necessary.",
        "Four successful node-update opportunities per structural action are matched. Actual changed nodes, entries, visits and L1/L2/Linf norms appear in per_seed.csv; successful no-op writes remain eligible.", "",
        "| Dataset | Budget | Arm | n | ACC (%) mean ± SD | 95% mean CI | Status |",
        "|---|---:|---|---:|---:|---|---|"]
    for row in report["summary"]:
        sd = "undefined" if row["sd"] is None else f"{row['sd']:.4f}"
        ci = "undefined" if row["ci95"] is None else f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]"
        lines.append(f"| {row['dataset']} | {row['budget']:g} | {row['variant']} | {row['n']} | {row['attacked_accuracy_percent_mean']:.4f} ± {sd} | {ci} | {'preliminary' if row['preliminary'] else 'five or more seeds'} |")
    lines += ["", "| Dataset | Budget | Paired contrast | Role | n | Difference (pp) | 95% mean CI | Holm p |",
              "|---|---:|---|---|---:|---:|---|---:|"]
    for row in report["contrasts"]:
        ci = "undefined" if row["ci95"] is None else f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]"
        p = "descriptive" if row["p_holm"] is None else f"{row['p_holm']:.4g}"
        lines.append(f"| {row['dataset']} | {row['budget']:g} | {row['contrast']} | {row['role']} | {row['n']} | {row['mean']:.4f} | {ci} | {p} |")
    if report["warnings"]:
        lines += ["", "## Warnings", "", *[f"- {value}" for value in report["warnings"]]]
    lines += ["", "See summary.json for all missing/rejected stages and paired_differences.csv for each seed's contrasts.",
              "Source/data/environment fingerprints are checked from the saved manifest; historical results do not need to match the current checkout. No manuscript is modified."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=["citeseer", "cora"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[15, 16, 17, 18, 19])
    parser.add_argument("--budgets", type=float, nargs="+", default=[.05, .1])
    args = parser.parse_args(argv)
    try:
        make_jobs(args.datasets, args.seeds, args.budgets)
    except ValueError as exc:
        parser.error(str(exc))
    out = args.output_dir or args.root.parent / "followup_reports/action_state"
    report = dict(protocol=PROTOCOL, expected_datasets=args.datasets, expected_seeds=args.seeds,
        expected_budgets=args.budgets, expected_blocks=len(args.datasets)*len(args.budgets)*len(args.seeds),
        missing=[], missing_jobs=[], missing_blocks=[], rejected_blocks=[], complete_blocks=[], warnings=[],
        summary=[], contrasts=[], paired_differences=[])
    per_seed = []
    for dataset in args.datasets:
        for budget in args.budgets:
            blocks = []
            for seed in args.seeds:
                jobs = make_jobs([dataset], [seed], [budget])
                rows, missing_variants = [], []
                for job in jobs:
                    path = args.root / dataset / f"ptb_{budget:g}" / f"seed_{seed}" / job["variant"] / "result.json"
                    if not path.is_file():
                        report["missing"].append(str(path))
                        report["missing_jobs"].append({**job, "path": str(path)})
                        missing_variants.append(job["variant"])
                        continue
                    try:
                        rows.append(validate_result(path, job))
                    except Exception as exc:
                        report["rejected_blocks"].append({**job, "path": str(path),
                            "error_type": type(exc).__name__, "error": str(exc)})
                if missing_variants:
                    report["missing_blocks"].append(dict(dataset=dataset, budget=budget, seed=seed, missing_variants=missing_variants))
                if len(rows) != len(jobs):
                    continue
                errors = validate_block(rows)
                if errors:
                    report["rejected_blocks"].append(dict(dataset=dataset, budget=budget, seed=seed, error=errors))
                    continue
                block = {row["variant"]: row for row in rows}
                blocks.append(block)
                report["complete_blocks"].append(dict(dataset=dataset, budget=budget, seed=seed))
                per_seed.extend(_per_seed(block[arm]) for arm in ARMS)
                for name, (first, second, role) in CONTRASTS.items():
                    difference = 100*(block[first]["evaluation"]["metrics"]["combined_accuracy"] - block[second]["evaluation"]["metrics"]["combined_accuracy"])
                    report["paired_differences"].append(dict(dataset=dataset, budget=budget, seed=seed,
                        contrast=name, role=role, first_arm=first, second_arm=second, difference_pp=difference))
            if not blocks:
                continue
            if len({_canonical(block["structure"]["provenance"]) for block in blocks}) != 1:
                report["warnings"].append(f"{dataset}/{budget:g}: mixed provenance across seeds; aggregates suppressed")
                continue
            seeds = [block["structure"]["seed"] for block in blocks]
            for arm in ARMS:
                stats = paired_statistics([100*block[arm]["evaluation"]["metrics"]["combined_accuracy"] for block in blocks])
                report["summary"].append(dict(dataset=dataset, budget=budget, variant=arm, seeds=seeds,
                    n=stats["n"], attacked_accuracy_percent_mean=stats["mean"], sd=stats["sd"], ci95=stats["ci95"], preliminary=stats["n"] < 5))
            for name, (first, second, role) in CONTRASTS.items():
                values = [100*(block[first]["evaluation"]["metrics"]["combined_accuracy"] - block[second]["evaluation"]["metrics"]["combined_accuracy"]) for block in blocks]
                report["contrasts"].append(dict(dataset=dataset, budget=budget, contrast=name, role=role,
                    seeds=seeds, **paired_statistics(values), preliminary=len(blocks) < 5, p_holm=None))
    planned = [dict(dataset=d, budget=b, contrast="pre_minus_post") for d in args.datasets for b in args.budgets]
    estimated = {(r["dataset"], r["budget"]): r for r in report["contrasts"] if r["role"] == "primary"}
    adjusted = holm([estimated.get((r["dataset"], r["budget"]), {}).get("p_signflip", 1.) for r in planned])
    for row, p in zip(planned, adjusted):
        if (row["dataset"], row["budget"]) in estimated:
            estimated[row["dataset"], row["budget"]]["p_holm"] = p
    report["holm_family"] = dict(size=len(planned), planned_contrasts=planned,
        missing_contrasts=[r for r in planned if (r["dataset"], r["budget"]) not in estimated],
        policy="Primary Pre-minus-Post only; missing requested cells padded with p=1 for Holm, never treated as measured results.")
    report["all_requested_blocks_complete"] = (len(report["complete_blocks"]) == report["expected_blocks"] and
        bool(report["summary"]) and not (report["missing"] or report["rejected_blocks"] or report["warnings"]))
    report["ready_for_full_seed_reporting"] = report["all_requested_blocks_complete"] and len(args.seeds) >= 5
    report["preliminary"] = any(row["preliminary"] for row in report["summary"])
    write_json(out / "summary.json", report)
    for filename, rows in (("per_seed", per_seed), ("accuracy_summary", report["summary"]),
                           ("paired_differences", report["paired_differences"]), ("paired_contrasts", report["contrasts"])):
        write_csv(out / f"{filename}.csv", rows)
    (out / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {out}; complete blocks={len(report['complete_blocks'])}; missing={len(report['missing'])}")
    return 0 if report["all_requested_blocks_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
