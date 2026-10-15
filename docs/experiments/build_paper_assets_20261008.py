"""Build the paper figure and a read-only audit from validated follow-up reports.

Run from any directory with the existing smc Python environment. No attacks run.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = (Path(__file__).resolve().parents[2]
        / "gbhaa_experiments/followup_reports/paper_2026-10-08")
EXPERIMENTS = HERE.parents[1]
PAPER = Path("/Users/shemingcong/Downloads/Thesis/GBHAA")
factorial = json.loads((HERE / "factorial/summary.json").read_text())
selectors = json.loads((HERE / "selectors/summary.json").read_text())
assert factorial["ready_for_full_seed_reporting"]
assert factorial["expected_seeds"] == [15, 16, 17, 18, 19]

variants = [["topology_ppr", "topology_ae_pr"],
            ["attribute_ppr", "attribute_ae_pr"]]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.88))
fig.subplots_adjust(left=.12, right=.99, bottom=.22, top=.82, wspace=.23)
for ax, budget in zip(axes, [.05, .1]):
    rows = {r["variant"]: r for r in factorial["summary"] if r["budget"] == budget}
    values = np.array([[rows[v]["attacked_accuracy_percent_mean"] for v in vs] for vs in variants])
    ax.imshow(values, vmin=56.5, vmax=60.5, cmap="Blues_r", aspect="equal")
    for i in range(2):
        for j in range(2):
            row = rows[variants[i][j]]
            best = values[i, j] == values.min()
            ax.text(j, i, f"{values[i,j]:.2f}\n± {row['sd']:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if values[i,j] < 58.5 else "#132b40",
                    weight="bold" if best else "normal", linespacing=1.4)
            if best:
                ax.add_patch(Rectangle((j-.48, i-.48), .96, .96, fill=False,
                                       edgecolor="#17212a", linewidth=1))
    ax.set_title(f"{budget:.0%} edge budget", fontsize=10, pad=8)
    ax.set_xticks([0, 1], ["L0", "L1"])
    ax.set_yticks([0, 1], ["G0", "G1"])
    ax.tick_params(length=0, pad=4)
    ax.set_xlabel("Local weighting", fontsize=8, labelpad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([-.5, .5, 1.5], minor=True)
    ax.set_yticks([-.5, .5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
stem = PAPER / "pictures/ae_pr_factorial_multiseed_pre_20261008"
fig.savefig(stem.with_suffix(".pdf"))
fig.savefig(stem.with_suffix(".png"), dpi=240)
plt.close(fig)

counts = []
for suite in ["factorial", "selectors"]:
    rows = list(csv.DictReader((HERE / suite / "per_seed.csv").open()))
    for budget in sorted({r["budget"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows}):
            block = [r for r in rows if r["budget"] == budget and r["variant"] == variant]
            entries = [int(r["changed_entries"]) for r in block]
            counts.append(dict(suite=suite, budget=budget, variant=variant,
                seeds=[int(r["seed"]) for r in block],
                changed_nodes=sorted({int(r["changed_nodes"]) for r in block}),
                node_visits=sorted({int(r["node_visits"]) for r in block}),
                entries_mean=statistics.mean(entries), entries_sd=statistics.stdev(entries),
                entries_min=min(entries), entries_max=max(entries)))

root = EXPERIMENTS / "followup_results_factorial_pre/factorial/citeseer"
rounds = [r for p in root.glob("*/seed_*/*/artifacts/rounds.json")
          for r in json.loads(p.read_text())["rounds"]]
rank_audit = {"round_count": len(rounds),
    "global_range_min": min(r["global_range"] for r in rounds),
    "global_range_max": max(r["global_range"] for r in rounds),
    "unique_scores_min": min(r["global_unique_scores"] for r in rounds),
    "unique_scores_max": max(r["global_unique_scores"] for r in rounds),
    "cutoff_ties_max": max(r["global_cutoff_tie_count"] for r in rounds)}
meta_sha = hashlib.sha256((EXPERIMENTS.parent / "meta.py").read_bytes()).hexdigest()
assert all(json.loads(p.read_text())["provenance"]["sources"]["meta.py"] == meta_sha
           for p in root.glob("*/seed_*/*/result.json"))
audit = dict(factorial_ready=True, selectors_ready=selectors["ready_for_full_seed_reporting"],
             factorial_rank_diagnostics=rank_audit, actual_modification_counts=counts,
             verified_meta_py_sha256=meta_sha,
             effective_citeseer_split="DeepRobust nettack (meta.py hard-coded; CLI normal is unused here)",
             final_feature_magnitude="Not recoverable: final feature matrices and norms were not saved")
(HERE / "paper_data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
with (HERE / "modification_counts.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(counts[0])); w.writeheader(); w.writerows(counts)
print("Figure:", stem.with_suffix(".pdf"))
print("Audit:", HERE / "paper_data_audit.json")
