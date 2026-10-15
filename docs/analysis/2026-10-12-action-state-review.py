"""Read verified results and write a separate analysis; never edit the manuscript."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "gbhaa_experiments/followup_reports/action_state_review_2026-10-12"
RESULTS = ROOT / "gbhaa_experiments/followup_results_action_state"
ARMS = ("parallel", "endpoint_pre", "endpoint_post")
NAMES = {"parallel": "Parallel", "endpoint_pre": "Endpoint-Pre", "endpoint_post": "Endpoint-Post"}
CELLS = [(d, b) for d in ("citeseer", "cora") for b in (.05, .1)]


def save_csv(name, rows):
    with (HERE/name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    report = json.loads((HERE/"summary.json").read_text())
    assert report["all_requested_blocks_complete"] and report["ready_for_full_seed_reporting"]
    assert len(report["complete_blocks"]) == 20
    assert not report["missing_jobs"] and not report["rejected_blocks"] and not report["warnings"]
    stats = {(r["dataset"], r["budget"], r["variant"]): r for r in report["summary"]}
    contrasts = {(r["dataset"], r["budget"], r["contrast"]): r for r in report["contrasts"]}
    counts, details, actual, probes = [], [], [], []
    for d, b in CELLS:
        records = {a: [] for a in ARMS}
        for seed in range(15, 20):
            row = dict(dataset=d, budget=b, seed=seed)
            for arm in ARMS:
                directory = RESULTS/d/f"ptb_{b:g}"/f"seed_{seed}"/arm
                result = json.loads((directory/"result.json").read_text())
                records[arm].append(result["attack"])
                with np.load(directory/"artifacts/combined_predictions.npz") as z:
                    acc = 100*float(np.mean(z["predictions"] == z["labels"]))
                assert np.isclose(acc, 100*result["evaluation"]["metrics"]["combined_accuracy"], atol=1e-10, rtol=0)
                row[arm] = acc
                rounds = json.loads((directory/"artifacts/rounds.json").read_text())["rounds"]
                probes.append(dict(dataset=d, budget=b, seed=seed, variant=arm,
                    rounds=len(rounds), changed_selection_rounds=sum(r["probe_jaccard"] < 1 for r in rounds),
                    changed_selection_fraction=float(np.mean([r["probe_jaccard"] < 1 for r in rounds])),
                    mean_jaccard=float(np.mean([r["probe_jaccard"] for r in rounds])),
                    mean_score_l1=float(np.mean([r["probe_score_l1"] for r in rounds]))))
            row["pre_minus_post_pp"] = row["endpoint_pre"]-row["endpoint_post"]
            row["parallel_minus_post_pp"] = row["parallel"]-row["endpoint_post"]
            details.append(row)
        for name in ("pre_minus_post", "parallel_minus_post"):
            values = np.array([r[name+"_pp"] for r in details if r["dataset"] == d and r["budget"] == b])
            st = contrasts[d, b, name]
            assert np.isclose(values.mean(), st["mean"], atol=1e-10, rtol=0)
            counts.append(dict(dataset=d, budget=b, contrast=name, mean_pp=st["mean"],
                sd_pp=st["sd"], ci_low=st["ci95"][0], ci_high=st["ci95"][1],
                post_wins=int((values > 1e-10).sum()), ties=int((abs(values) <= 1e-10).sum()),
                post_losses=int((values < -1e-10).sum()), p_signflip=st["p_signflip"], p_holm=st["p_holm"]))
        for arm in ARMS:
            row = dict(dataset=d, budget=b, variant=arm)
            for key in ("completed_edges", "attribute_node_visits", "changed_feature_nodes",
                        "changed_feature_entries", "feature_l1", "feature_l2", "feature_linf"):
                row[key+"_mean"] = float(np.mean([r[key] for r in records[arm]]))
            actual.append(row)
    save_csv("seed_accuracy_comparison.csv", details)
    save_csv("stability_summary.csv", counts)
    save_csv("actual_perturbation_summary.csv", actual)
    save_csv("within_state_probes.csv", probes)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True, sharey=True)
    labels = [f"{d.title()}  {b:.0%}" for d, b in CELLS]
    for ax, name, title in zip(axes, ("pre_minus_post", "parallel_minus_post"),
                               ("Endpoint-Pre minus Endpoint-Post", "Parallel minus Endpoint-Post")):
        ax.axvline(0, color="#8c9199", linestyle="--", linewidth=1)
        for i, (d, b) in enumerate(CELLS):
            st = contrasts[d, b, name]; low, high = st["ci95"]
            color = "#087f5b" if low > 0 else "#c0392b" if high < 0 else "#54657a"
            ax.errorbar(st["mean"], i, xerr=[[st["mean"]-low], [high-st["mean"]]],
                        fmt="o", color=color, capsize=4, markersize=7, linewidth=2)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_yticks(range(4), labels=labels)
        ax.set_xlim(-3.65, 1.9); ax.set_xlabel("Paired ACC difference (percentage points)")
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    axes[0].invert_yaxis(); axes[1].tick_params(labelleft=True)
    fig.suptitle("Endpoint-Post: no consistent advantage across conditions", fontsize=14, y=.98)
    fig.text(.5, .035, "Positive favors Post. Dots: paired means; bars: unadjusted t-based 95% CIs, n = 5 seeds.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(HERE/"paired_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = ["# 三组 matched action-state 实验审查（2026-10-12）", "",
        "## 结论", "",
        "默认实验全部完整：20/20 配对种子区组，20/20 共享结构轨迹和 60/60 三组攻击均通过文件级校验。",
        "Endpoint-Post 相比 Endpoint-Pre 没有跨数据集、跨预算的稳定优势；相比 Parallel，仅 Cora 5% 方向一致有利，Citeseer 10% 则方向一致不利。",
        "未修改论文正文或实验原始数据；Pubmed 是可选扩展，不计入本次默认完成率。", "",
        "## 完整性与配对", "",
        "范围：Citeseer / Cora × 5% / 10% × seeds 15–19。缺失、被拒绝记录和配对警告均为 0。",
        "三组在每一区组内共享结构动作、最终图、原始特征、训练/评估节点、clean predictions 和 victim seed；全部记录使用 degree 全局源，来源文件和环境一致。",
        "校验包括图的逐步回放、每轮四节点更新、访问次数、实际特征写入、预测及 ACC 重算、完整请求缓存的请求/响应哈希。",
        "另独立从 60 份 combined_predictions.npz 重算准确率，并检查配对的标签、节点和 clean predictions。", "",
        "## 被攻击 ACC（%，越低攻击越强）", "",
        "| 数据集 | 预算 | Parallel | Endpoint-Pre | Endpoint-Post |",
        "|---|---:|---:|---:|---:|"]
    for d, b in CELLS:
        vals = [f"{stats[d,b,a]['attacked_accuracy_percent_mean']:.2f} ± {stats[d,b,a]['sd']:.2f}" for a in ARMS]
        lines.append(f"| {d.title()} | {b:.0%} | " + " | ".join(vals) + " |")
    lines += ["", "均值 ± 样本标准差；每格 5 个配对种子。各组均值的 CI 见 accuracy_summary.csv。", "",
        "## 配对效果", "",
        "差值均为“对照 ACC − Post ACC”，单位为百分点；正值有利于 Post。胜/平/负也从 Post 的角度计数。", "",
        "| 数据集 | 预算 | 对照 | 平均差值 | 95% CI | Post 胜/平/负 |",
        "|---|---:|---|---:|---|---|"]
    for r in counts:
        control = "Endpoint-Pre" if r["contrast"] == "pre_minus_post" else "Parallel"
        lines.append(f"| {r['dataset'].title()} | {r['budget']:.0%} | {control} | {r['mean_pp']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['post_wins']}/{r['ties']}/{r['post_losses']} |")
    lines += ["", "主比较 Pre−Post：四个条件的 CI 均包含 0，且平均差值有正有负。当前结果不足以支持修改后图必然带来更有效的属性定位；也不能由此认定两者等效。",
        "辅助比较：Cora 5% 的 Post 在 5/5 种子上优于 Parallel，平均改善 0.733 pp；Citeseer 10% 的 Post 在 5/5 种子上更差，平均劣化 1.833 pp。",
        "这些 CI 是各条件独立计算的、未经多重校正的 t 区间。五种子双侧精确符号翻转检验最小 p 为 0.0625；上述两项的精确 p 均为 0.0625。因此不把某个 t 区间不跨 0 写成无条件的稳健显著结论。",
        "主比较的四条件 Holm 校正 p 均为 0.75。分析单位是 seed，没有把所有节点或各预算的同名 seed 当成额外独立重复。", "",
        f"![配对差值和置信区间]({HERE/'paired_effects.png'})", "",
        "## 机制诊断", "",
        "在 Endpoint-Post 每轮的同一特征快照、相同资格与端点上，比较改边前后扩散探针：", "",
        "| 数据集 | 预算 | Pre/Post 选中集合发生变化的轮次 | 平均 Jaccard |",
        "|---|---:|---:|---:|"]
    for d, b in CELLS:
        p = [r for r in probes if r["dataset"] == d and r["budget"] == b and r["variant"] == "endpoint_post"]
        changed = sum(r["changed_selection_rounds"] for r in p); total = sum(r["rounds"] for r in p)
        lines.append(f"| {d.title()} | {b:.0%} | {changed}/{total}（{changed/total:.2%}） | {np.mean([r['mean_jaccard'] for r in p]):.4f} |")
    lines += ["", "这些描述性探针说明：改边确实能改变扩散和选点，但本次约 82%–84% 的轮次仍选中同一集合。它们没有证明这种变化带来更强攻击，也不能作为大量独立样本做显著性检验。",
        "各组累计特征会分化，最终 ACC 体现固定结构轨迹下的累积策略差异；探针才是在单一特征状态上隔离 pre/post 图差异。", "",
        "## 实际扰动量", "",
        "每轮四次节点更新机会匹配，实际唯一节点数与修改条目数并非强制相同。下表为五种子均值：", "",
        "| 数据集 | 预算 | 组别 | 节点更新次数 | 实际修改节点 | 实际修改条目 | L1 | L2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in actual:
        lines.append(f"| {r['dataset'].title()} | {r['budget']:.0%} | {NAMES[r['variant']]} | {r['attribute_node_visits_mean']:.0f} | {r['changed_feature_nodes_mean']:.1f} | {r['changed_feature_entries_mean']:.1f} | {r['feature_l1_mean']:.1f} | {r['feature_l2_mean']:.2f} |")
    lines += ["", "Parallel 的实际修改覆盖更广，这属于选择策略造成的结果差异，可能参与解释其攻击效果；不能把对比说成完全匹配了最终唯一节点数或 L1 修改量，也不能仅凭此表断定因果解释。",
        "全部三组使用记录中的真实短词表模式（partial vocabulary），没有 fallback 词表。结论适用于本次固定四节点/轮、相同生成投影与 GCN victim 的协议。", "",
        "## 可据此表述的结论", "",
        "> 在共享结构轨迹和匹配节点更新预算下，修改后的扩散图会改变部分轮次的属性定位；但在 Citeseer 与 Cora、5% 与 10% 预算及 seeds 15–19 上，Endpoint-Post 相比 Endpoint-Pre 未显示稳定的攻击优势。相对独立重启的 Parallel 对照，其效果依赖数据集与预算。", "",
        "不建议为了填补缺失重跑：本次默认实验已经完整。若后续研究差异来源，应先明确新的机制假设，再预先规划更多种子或另一种预算定义，避免根据结果反复调整配置。", "",
        "## 文件", "",
        "- summary.json / summary.md：原协议完整性审查和统计汇总。",
        "- seed_accuracy_comparison.csv：每种子的三组 ACC 与两个 Post 对比。",
        "- stability_summary.csv：配对效应、区间、胜平负。",
        "- actual_perturbation_summary.csv：实际修改量。",
        "- within_state_probes.csv：各组、各种子的同状态 pre/post 探针。",
        f"- [可复现生成脚本]({Path(__file__).resolve()})：本报告和图的分析源码。", ""]
    (HERE/"analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")
    paper = Path("/Users/shemingcong/Downloads/Thesis/GBHAA/Main-AAAI.tex")
    print("paper_sha256", hashlib.sha256(paper.read_bytes()).hexdigest())
    print("Wrote", HERE/"analysis_zh.md")


if __name__ == "__main__":
    main()
