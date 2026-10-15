# Post 选点机制快速实验

## 为什么目前没有稳定优势

证据来自已完成的 Citeseer 确认实验，seeds 16–19，每个种子 120 次结构动作。
原始诊断见 `followup_reports/post_refinement_confirm/selector_mechanism_diagnosis.json`。

1. **端点占据一半属性预算。** α=.50 的 Post 四个种子共 480 轮，每轮都选中两个结构端点；α=.28 的 baseline Post 也如此。每轮只写四个节点，因此扩散到邻域的选择只剩两个名额。Pre/Post 使用同一端点，所以这个共同项容易压过翻边带来的排名变化。
2. **图变了，大部分选点集合没有变。** 在 α=.50 Post 的每轮相同属性状态上重算 Pre/Post，只有 43/480 轮（8.96%）改变四节点集合；α=.28 baseline Post 为 81/480 轮（16.88%）。这支持“提高重启概率进一步压缩了对图变化的响应”，但不能仅凭相关性判定 ACC 的原因。这里没有把不同运行已经分叉的属性状态当成同一状态。baseline Pre 未包含在这批确认实验内。
3. **属性相似度为零时，翻边无法改变当前加权传播。** α=.50 Post 的 85/480 轮，动作两端的当前属性余弦为零；在 floor=0 的算子中，这条新增或删除边权重为零。这些轮次不能通过该扩散算子体现动作差异。已有 floor 微调也没有显示稳定收益，因此本次固定 floor=0，单独检验选点机制；`frontier` 与 `balanced_frontier` 在这种情况下也无法改变扩散排名；`frontier_editgain` 的 GCN 梯度使用该组未乘余弦权重的原始图，所以翻边仍可能改变潜力分数，但不保证改变选点或 ACC。
4. **生成步骤可能弱化选点优势。** 480 轮都只有一个四节点文本组。当前 `_build_round_text_clusters` 在节点数不超过 `text_max_cluster_size=4` 时直接合为一组，计算出的图嵌入并不参与进一步分组。所选四节点的平均两两余弦从约 .191 上升到 .832。代码未直接优化分类损失；因此“扩散排名高”不等于“写入这批文本后更能破坏分类”。余弦变化是观测事实，不单独证明同质化导致 ACC 上升。
5. **Parallel 覆盖更广。** α=.50 Post 各种子覆盖 379/366/389/369 个不同节点，Parallel 为 410/418/410/422 个。相同 480 次写入，后者平均多覆盖约 39 个节点。这是另一个可能的抵消因素，不能据此预设 Parallel 必然更好。

已有 α=.50 Post 相对 Pre/Parallel 的四种子平均收益都为负，配对 95% 区间都跨零；相对 α=.28 baseline Post，四个种子都更差。因此不把 α=.50 继续视为已证实的赢家。本次 α 固定 .28，目标是比较新选点规则是否改善实际验证集结果。

## 新增四配置 × 三组对照

| 配置 | 选点规则 | 检验的问题 |
|---|---|---|
| `baseline` | 原始双源扩散排名 | 相同代码、预算和种子下的参照 |
| `frontier` | 优先选择具有正扩散质量的合法非重启节点 | 把属性预算从端点转移到可达邻域是否更有效 |
| `balanced_frontier` | 两个单源扩散分别排名，交替取点并去重 | 防止某一端的邻域占据全部名额是否有益 |
| `frontier_editgain` | 同图扩散前 32 个合法非重启候选，再按代理模型的属性修改潜力排名 | 将局部性与分类脆弱性结合是否更有效 |

所有配置固定 α=.28、60 次扩散、floor=0、每轮四次节点写入、每节点最多十次访问；生成、投影和 victim 训练规则沿用原方案。没有改 `heir.py` 的主循环。

每个配置都运行三组：

- **Endpoint-Post**：执行该次翻边后的图，真实两个端点重启。
- **Endpoint-Pre**：翻边前的图，同一真实端点重启。
- **Parallel**：翻边前的图，按种子/轮次独立选定两个重启节点。

各组从同一初始状态回放相同结构动作前缀，此后各自累积属性修改。所有组用于生成的分组图仍是 Post 图。候选来自各自的扩散图，Pre/Parallel 不共享 Post 筛出的候选池。“非重启节点”在 Parallel 中排除的是独立重启节点，而非真实结构端点。

`frontier`/`balanced_frontier` 若不足四个有正质量的合法非重启节点，按原始扩散排名补足，可能包含重启节点；达到十次访问的节点始终不入选。双源策略按重启节点 ID 确定交替顺序，交换无向边两端的输入顺序不改变结果。

`frontier_editgain` 使用已有干净 surrogate，不额外拟合。伪标签在回放前固定为干净预测 argmax；梯度目标是全部非训练节点，包括不属于 train/validation/test 的未分配节点。**不使用非训练节点的真标签、验证 ACC 或测试 ACC 进行选点。** 同一配置三组使用相同代理模型、伪标签和目标节点。梯度前向传播使用该组的原始二值图，经 GCN 自环及对称归一化；它与前置候选池的余弦加权扩散是两个明确不同的步骤。

其潜力分数是：可使伪标签 NLL 增大的删除贡献 `sum(max(-g*x,0))`，加上词表内当前不存在的词中最多 70 项正梯度（假定单位添加）。候选先按扩散截到最多 32 个，再按潜力、扩散分数、节点 ID 排序。潜力全零时退回 frontier。这个分数仅为一阶机会估计；LLM 生成及相似度投影可能无法实现这些修改，**不是实际攻击增益或保证 Post 领先的指标**。这是带额外代理模型信息的方法变体，不能直接将其结果当作原始 AE-PR 的结果。

## 运行

在 `my_attack` 根目录、原有 `(smc)` 环境运行。

```bash
# 预览：不会训练、调用 LLM 或创建实验结果目录
bash gbhaa_experiments/launchers/run_post_mechanism.sh

# 推荐：Citeseer、seeds 15/16、每次 60 个结构动作，共 24 个短试验
bash gbhaa_experiments/launchers/run_post_mechanism.sh --execute

# 完成后汇总
bash gbhaa_experiments/launchers/run_post_mechanism.sh summary
```

如果只想先跑一个种子，以下命令共 **12 个短试验**，使用单独目录，避免随后扩成双种子时计划指纹冲突：

```bash
bash gbhaa_experiments/launchers/run_post_mechanism.sh --seeds 15 \
  --output-root gbhaa_experiments/post_mechanism_seed15_results --execute
bash gbhaa_experiments/launchers/run_post_mechanism.sh summary \
  --root gbhaa_experiments/post_mechanism_seed15_results \
  --output-dir gbhaa_experiments/followup_reports/post_mechanism_seed15
```

程序自动读取已有 `followup_results_action_state/.../structure` 的结构轨迹。`source-budget=0.1` 是轨迹来源；只回放前 60 次动作，Citeseer 实际结构扰动率约 **1.636%**，不是完整 10% 实验。一次运行失败后，保持全部参数相同，加 `--retry-failed` 可归档失败尝试后继续；若改代码、环境或任务矩阵，则使用新的 `--output-root`。

默认新结果：`gbhaa_experiments/post_mechanism_results`。
默认报告：`gbhaa_experiments/followup_reports/post_mechanism`。
旧实验目录和论文正文没有修改。

## 跑完如何判断

- `summary.md` / `ranking.csv`：比较 Post 的验证集 attacked ACC（越低攻击越强），不能只比较排序分数。
- `paired_comparisons.csv`：同一配置内 Post 对 Pre/Parallel；`control_minus_post_accuracy_pp_mean > 0` 才是 Post 有利。另有新 Post 对 baseline Post 的比较。
- `per_trial.csv`：实际修改节点/条目/幅度、覆盖率、端点占用率、Pre/Post 探针改变选点的频率，以及新策略回退比例。
- `rounds.json` 的 `mechanism_traces`：保存每个图的扩散状态、候选池、选点、潜力和回退信息。汇总会按当轮属性和图重新计算扩散并检查排名；同时检查代理模型身份、固定伪标签及目标集。审计不重新训练代理模型或独立重算梯度，梯度潜力属于保存的运行证据。

此次是两个已用于开发的种子上的机制筛选，不是独立确认，也不足以证明稳定优势。选定方案后需要冻结配置，用额外种子及更长轨迹复查；如果只增加选点差异但 ACC 不改善，不能据此宣称 Post 的机制得到验证。完整消融结论还需要相应实验支持。
