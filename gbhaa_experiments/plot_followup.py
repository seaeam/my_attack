"""Render checked multi-seed follow-up summaries to a separate figure directory."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def validate_report(report):
    if not report.get("ready_for_full_seed_reporting"):
        raise ValueError("Requested matrix is incomplete or failed verification")
    rows = report.get("summary", [])
    if not rows or any(row["n"] < 2 or row["sd"] is None for row in rows):
        raise ValueError(
            "Multi-seed plots require at least two complete seeds; a pilot is not sufficient"
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("gbhaa_experiments/followup_reports/new/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gbhaa_experiments/followup_reports/new/figures"),
    )
    args = parser.parse_args(argv)
    rows = validate_report(json.loads(args.summary.read_text()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = {}
    for row in rows:
        groups.setdefault((row["suite"], row["budget"]), []).append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "ae_pr": "Endpoint AE-PR",
        "pr": "PageRank (PR)",
        "ppr": "Endpoint PPR",
        "degree": "Degree",
        "random": "Random",
        "pre": "Pre-flip localization",
        "post": "Post-flip localization",
    }
    for (suite, budget), group in groups.items():
        n = group[0]["n"]
        if suite == "factorial":
            lookup = {r["variant"]: r for r in group}
            order = [
                ["topology_ppr", "topology_ae_pr"],
                ["attribute_ppr", "attribute_ae_pr"],
            ]
            values = np.array(
                [
                    [lookup[v]["attacked_accuracy_percent_mean"] for v in row]
                    for row in order
                ]
            )
            fig, ax = plt.subplots(figsize=(5.8, 4.4), layout="constrained")
            im = ax.imshow(values, cmap="Blues_r")
            for i in range(2):
                for j in range(2):
                    r = lookup[order[i][j]]
                    ax.text(
                        j,
                        i,
                        f"{values[i,j]:.2f} ± {r['sd']:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
            ax.set_xticks([0, 1], ["PPR", "AE-PR"])
            ax.set_yticks([0, 1], ["Topology weights", "Attribute weights"])
            ax.set_xlabel("Local endpoint-conditioned diffusion")
            ax.set_ylabel("Global degree-source accumulation")
            fig.colorbar(im, ax=ax, label="Attacked accuracy (%) ↓")
        else:
            fig, ax = plt.subplots(
                figsize=(max(5.8, len(group) * 1.35), 4.2), layout="constrained"
            )
            means = [r["attacked_accuracy_percent_mean"] for r in group]
            ax.bar(
                range(len(group)),
                means,
                yerr=[r["sd"] for r in group],
                capsize=4,
                color="#346b94",
            )
            ax.set_xticks(
                range(len(group)),
                [
                    labels.get(r["variant"], r["variant"].replace("_", "\n"))
                    for r in group
                ],
            )
            ax.set_ylabel("Attacked accuracy (%) ↓")
            ax.set_ylim(0, 100)
            for i, mean in enumerate(means):
                ax.text(i, mean + group[i]["sd"] + 1, f"{mean:.2f}", ha="center")
            ax.spines[["top", "right"]].set_visible(False)
        dataset = "Cora" if suite == "uniform_repair" else "Citeseer"
        ax.set_title(
            f"{dataset} · {suite} · {100*budget:g}% edges\nmean ± SD; n={n} paired seeds; retrained GCN"
        )
        stem = args.output_dir / f"{suite}_{budget:g}"
        fig.savefig(str(stem) + ".png", dpi=250)
        fig.savefig(str(stem) + ".svg")
        plt.close(fig)
    print(
        f"Saved {len(groups)} plots to {args.output_dir}; manuscript figures were not changed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
