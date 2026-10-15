# Edit-gain 重复选点与覆盖快速对比

## 这次具体解决什么

旧 `post_mechanism_results` 中，Citeseer seeds 15/16 的 `frontier_editgain` Post 各有 240 次节点写入，却只覆盖 113/121 个不同节点。两种子合计重复写入占 51.25%。

重新检查保存的逐轮候选池发现：32 节点池平均还有 26.52/26.55 个从未访问过的节点；120 轮中有 53 轮，池内至少有两个新节点，实际却选了不到两个。重复节点最高潜力与新节点最高潜力之比的中位数约为 2.11/2.08。因此这次保留原来的候选池，只改变池内排名，检验原始潜力偏向重复节点是否抵消了 Post 的收益。

这是待检验的解释。覆盖增加不等于 attacked ACC 一定下降；旧结果也没有证明 Post 同时优于 Pre 和 Parallel。

## 四种配置

| 配置 | 规则 |
|---|---|
| `baseline` | 原始扩散排名，作为参照 |
| `frontier_editgain` | 原来最多 32 个局部候选，按原始属性修改潜力排序 |
| `frontier_editgain_decay` | 同一候选池，按 `原始潜力 / (1 + 此前访问次数)` 排序，逐渐降低重复节点优先级 |
| `frontier_editgain_fresh3` | 同一候选池，先按原始潜力保留最多三个从未访问过的节点，再按原始潜力补齐四个名额 |

池内新节点充足时，`fresh3` 保证至少选三个新节点，第四个也可以是新节点。若池内新节点不足三个，就保留全部可用新节点，记录配额短缺，再按原有排名补齐，始终保持每轮四次写入。配额只作用于原始候选池，不扩大池，也不从池外寻找高潜力节点。池不足四点时沿用原有 frontier/合法扩散回退。

所有增益为零时，`decay` 按扩散分数和节点 ID 排序；`fresh3` 仍预留新节点名额，在各组内按扩散分数和 ID 排序。日志中的 `zero_gain_fallback=true` 表示池内没有正增益，对 `fresh3` 不表示取消新节点配额。相同分数使用固定 ID 顺序；访问达到十次的节点不能再次入选。第一轮全是新节点时，两种新规则与原 editgain 排序一致。

## 三组严格匹配

每种规则都分别用于三组，而非只给 Post 改善覆盖：

- **Endpoint-Post**：真实结构端点，在当次翻边后的图上扩散、计算潜力。
- **Endpoint-Pre**：相同真实端点，在当次翻边前的图上扩散、计算潜力。
- **Parallel**：按种子和轮次独立确定两个重启节点，在翻边前的图上扩散、计算潜力。

各组使用自己的属性状态、候选池和访问记录。结构动作前缀、初始图/属性、代理模型、固定伪标签、梯度目标集、生成/投影和 victim 协议一致。生成时的分组图三组均沿用 Post 图。属性状态在各自回放中可以逐渐分叉，不强制共享修改后的属性。

固定 α=.28、扩散 60 次、floor=0、候选池最多 32 点、每轮四次写入、每点最多十次访问。没有增加 LLM 调用预算，没有改变原来的 `heir.py` 主循环。潜力仍来自固定干净代理模型的伪标签 NLL 梯度，不使用非训练节点的真标签选点；它只是修改机会估计，不保证生成和投影能实现相应收益。

## 运行命令

在 `my_attack` 根目录、原有 `(smc)` 环境执行：

```bash
# 只预览，不训练、不调用 LLM、不创建结果目录
bash gbhaa_experiments/launchers/run_post_coverage.sh

# 默认：Citeseer，seeds 15/16，60 个结构动作，4 配置 × 3 组 × 2 种子 = 24 个短试验
bash gbhaa_experiments/launchers/run_post_coverage.sh --execute

# 跑完汇总
bash gbhaa_experiments/launchers/run_post_coverage.sh summary
```

更快的单种子版本共 12 个短试验，使用单独目录，避免以后扩成双种子时计划冲突：

```bash
bash gbhaa_experiments/launchers/run_post_coverage.sh --seeds 15 \
  --output-root gbhaa_experiments/post_coverage_seed15_results --execute
bash gbhaa_experiments/launchers/run_post_coverage.sh summary \
  --root gbhaa_experiments/post_coverage_seed15_results \
  --output-dir gbhaa_experiments/followup_reports/post_coverage_seed15
```

默认结果在 `gbhaa_experiments/post_coverage_results`，默认报告在 `gbhaa_experiments/followup_reports/post_coverage`。这次 baseline 和原 editgain 也在同一代码版本重新运行，避免混合旧运行指纹；旧结果目录保留。

程序读取已有 action-state 结构轨迹。`source-budget=.1` 指轨迹来源，实际只回放 60 个动作；Citeseer 有效结构扰动率约为 1.636%，不能解释为完整 10% 攻击。失败后保持参数、代码和环境不变，可加 `--retry-failed` 归档失败尝试并继续；代码/环境/任务矩阵变化时另设新的 `--output-root`。

## 报告与判定

- `summary.md` / `ranking.csv`：验证集 attacked ACC 均值与样本标准差，越低表示攻击越强。
- `paired_comparisons.csv`：新 Post 对原 editgain Post、baseline Post，以及同一新规则的 Pre/Parallel。`control_minus_post_accuracy_pp_mean > 0` 表示 Post 有利；保留每个种子差值、ACC 胜/负/平次数、样本标准差和配对 95% t 区间。
- `per_trial.csv`：最终覆盖节点数、首次/重复写入比例、实际修改节点/条目/幅度；新规则额外记录每轮池内新节点数、选中新节点数和 `fresh3` 配额短缺。配额短缺是池内不足三个新节点，不等于最终没有补满四次写入。
- `rounds.json`：保存池、原始/调整潜力、所选节点的此前访问次数和新节点配额。汇总会按保存的真实写入重建访问历史并重新计算扩散/排名，拒绝不一致的诊断；检查固定代理模型、伪标签和目标集身份。不会重新拟合代理模型或独立重算已保存梯度。

首先检查新 Post 是否改善原 editgain 的 ACC 和重复率，再检查是否同时胜过同规则 Pre/Parallel。不能把覆盖提高或 NLL 单独提高当成 ACC 胜出。两种子短轨迹属于开发筛选，不能证明稳定优势；若观察到一致收益，冻结配置后再用更多种子和更长轨迹确认。本次未修改论文正文。
