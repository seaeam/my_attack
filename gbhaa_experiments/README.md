# GBHA 实验套件

本目录包含针对 GBHA 论文七项实证要求编写的独立实验代码。实验变体均放在单独目录中，不改变原有 `meta.py` 和 `heir.py` 的标准运行入口。

完整的实验补充方案、推荐运行顺序、烟测命令、结果汇总和论文表格设计见 [`EXPERIMENT_SUPPLEMENT_PLAN_ZH.md`](./EXPERIMENT_SUPPLEMENT_PLAN_ZH.md)。

理论要求 **GB sensitivity preservation（GB 敏感性保持）** 本身不是可直接运行的实验。其经验验证由 `coarse_gradient_consistency.py` 完成，用于衡量粗粒度球对梯度是否保留细粒度节点对梯度的排序。

## 目录结构

```text
gbhaa_experiments/
├── run_matrix.py                    # 按数据集、随机种子和预算运行实验矩阵
├── specs.py                         # 实验及对照变体定义
├── runner.py                        # 运行计时、峰值内存统计和输出解析
├── coarse_gradient_consistency.py   # 粗粒度与细粒度梯度一致性
├── stealthiness.py                  # 结构与特征空间隐蔽性指标
├── run_stealthiness.py              # 生成攻击产物并执行隐蔽性分析
├── aggregate_results.py             # 跨随机种子计算均值和标准差
├── entrypoints/                     # 独立实验攻击入口
├── launchers/                       # 每项实验对应的启动脚本
└── tests/                           # 实验框架单元测试
```

实验结果统一写入 `gbhaa_experiments/results/`。每次实际运行会保存：

- 完整的标准输出和错误日志；
- 已隐藏 API Key 的实验命令；
- 总运行时间；
- 系统支持时由 `/usr/bin/time` 记录的峰值常驻内存；
- 操作系统、Python 版本、CPU 数量和可见 CUDA 设备；
- 系统存在 `nvidia-smi` 时记录 GPU 型号和驱动版本；
- 干净图、结构攻击、特征攻击和联合攻击准确率；
- 数据集、攻击预算、随机种子和实验变体信息。

## 执行策略

所有启动脚本默认只执行 **dry-run（命令预览）**。完整实验矩阵包含大量 GNN 训练和 LLM 请求，因此应先检查打印出的命令，再设置 `EXECUTE=1` 正式运行：

```bash
cd /Users/bytedance/Downloads/Github/my_attack
EXECUTE=1 bash gbhaa_experiments/launchers/run_gb_ablation.sh
```

如果项目使用的 Python 环境不是默认环境，可通过 `PYTHON_BIN` 指定：

```bash
PYTHON_BIN=/path/to/python EXECUTE=1 \
  bash gbhaa_experiments/launchers/run_efficiency.sh
```

无需修改脚本即可覆盖默认数据集：

```bash
DATASETS="citeseer" EXECUTE=1 \
  bash gbhaa_experiments/launchers/run_hybrid_ablation.sh
```

需要外部模型的实验默认连接本地 Ollama 兼容接口。可通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OLLAMA_MODEL` 切换服务或模型。实验元数据不会保存明文 API Key。

## Citeseer 特征词表说明

本套件中所有小图实验统一使用 `citeseer`，不使用其他小图数据集。

Citeseer 节点特征为 3,703 维，但仓库现有 Citeseer 缓存词表只有 500 个词，二者不存在完整的一一对应关系。因此，文本相关启动脚本会显式启用 3,703 个 `feature_i` 占位特征词，使每个占位词与一列节点特征严格对应。

这意味着：

- Citeseer 上的相关实验属于离散特征空间中的 LLM 辅助扰动；
- 不能将 `feature_i` 解释成真实单词；
- 不能将实验表述为自然语言语义攻击；
- 不显式允许占位特征词时，词表维数不匹配会直接导致实验停止；
- 是否使用占位特征词会记录在命令和运行日志中。

## 1. 大规模效率实验

```bash
bash gbhaa_experiments/launchers/run_efficiency.sh
```

默认数据集为 Citeseer、Pubmed 和 CS，攻击预算为 1%、5% 和 10%，随机种子为 15、16、17、18、19。

实验变体：

- `gb`：Granular-Ball 层次结构搜索；
- `kmeans`：使用相同聚类数量的 K-Means 层次搜索；
- `node_level`：设置 `--level 1`，将候选细化为单节点，作为无粗化参考。

`node_level` 只在 Citeseer 上运行。Pubmed 和 CS 只比较 GB 与 K-Means。

该实验只执行结构攻击，不启用属性或 LLM 攻击，避免模型调用延迟掩盖结构搜索成本。主要记录：

- Edge Accuracy；
- 相对干净图的准确率下降；
- 总运行时间；
- 峰值内存；
- 实际完成的结构扰动数量。

CS 需要项目环境已经安装 PyG 相关依赖。当前不默认运行 `ogbn-arxiv`，因为原始数据加载路径尚未经过完整可执行验证。

## 2. Granular-Ball 消融实验

```bash
bash gbhaa_experiments/launchers/run_gb_ablation.sh
```

默认数据集为 Citeseer 和 Pubmed，攻击预算为 5% 和 10%，随机种子为 15、16、17、18、19。

实验变体：

- `gb`：Granular-Ball 划分；
- `kmeans`：K-Means 划分；
- `random`：在相同目标簇数下进行均衡随机划分；
- `node_level`：无粗化节点级搜索，只在 Citeseer 上运行。

所有变体使用相同的：

- 攻击预算；
- 随机种子；
- 代理模型；
- 层次深度；
- 内部优化设置；
- 全局重要节点配置。

该实验只改变分组和搜索骨架，主要指标为 Edge Accuracy，同时记录运行时间和峰值内存。

## 3. Hybrid Coupling 消融实验

```bash
bash gbhaa_experiments/launchers/run_hybrid_ablation.sh
```

默认数据集为 Citeseer，攻击预算为 5% 和 10%，随机种子为 15、16、17、18、19。

| 实验变体 | 实验含义 | 主要指标 |
|---|---|---|
| `edge_only` | 只修改图结构 | Edge Accuracy |
| `feature_only` | 邻接矩阵全程固定，只执行独立特征梯度攻击 | Feature Accuracy |
| `parallel` | 结构目标和属性目标独立选择，再联合评估 | Combined Accuracy |
| `serial` | 结构端点引导局部属性候选选择 | Combined Accuracy |

`feature_only` 与 `parallel` 使用相同的独立特征梯度种子机制，但两者不是同一次攻击：

- `feature_only` 从开始到结束不搜索或翻转边；
- `parallel` 会执行匹配的结构扰动；
- 两者分别运行并产生不同的攻击产物。

每次完整属性攻击会分别评估：

- 干净图与原始特征；
- 扰动图与原始特征；
- 原始图与扰动特征；
- 扰动图与扰动特征。

## 4. LLM 消融实验

```bash
bash gbhaa_experiments/launchers/run_llm_ablation.sh
```

默认数据集为 Citeseer，攻击预算为 5% 和 10%，随机种子为 15、16、17、18、19。

实验变体：

- `full`：完整的结构—特征联合攻击；
- `deterministic_no_llm`：使用特征维数对齐的确定性占位模板，不调用外部模型或 API；
- `without_discriminative_words`：设置 `text_cdl_topk=0`，移除类别判别特征词；
- `without_keyword_preservation`：设置 `text_budget_per_node=0`，移除原始特征词保留；
- `without_similarity_projection`：设置 `text_similarity_min=0`，取消相似度投影；
- `without_added_word_cap`：将新增特征词上限设为一个实际不会触发的数值。

`deterministic_no_llm` 使用相同的 3,703 维占位特征词和候选选择流程，但外部模型/API 调用次数为 0。

所有变体都属于 Citeseer 离散特征空间消融，不能据此主张自然语言语义保持或文本流畅性。

## 5. 粗粒度梯度一致性实验

```bash
bash gbhaa_experiments/launchers/run_coarse_gradient_consistency.sh
```

默认配置：

- 数据集：Citeseer；
- 随机种子：15；
- 粒球数量：8、16、32；
- Top-k：5。

对每个粒球数量，实验依次执行：

1. 在原始图上训练相同的内部攻击模型；
2. 计算完整节点级邻接矩阵梯度；
3. 建立由全局重要节点引导的 GB 划分；
4. 计算粗化图上的球对梯度；
5. 将每个粗粒度球对映射到其中最优的可行细粒度节点对；
6. 比较粗粒度排序与细粒度排序。

输出指标包括：

- Spearman 相关系数；
- Kendall 相关系数；
- 加边/删边动作一致率；
- Top-k 重合率；
- Top-1 命中；
- Search Regret。

细粒度参考是完整图上的一阶梯度，不是对每条候选边实际翻转后重新训练得到的穷举非线性结果。

该实验使用稠密节点级邻接梯度，因此只适合小图。`--max-nodes` 仅用于烟测和调试，不能作为论文正式结果。

## 6. AE-PPR 对比实验

```bash
bash gbhaa_experiments/launchers/run_ae_ppr_comparison.sh
```

默认数据集为 Citeseer，攻击预算为 5% 和 10%，随机种子为 15、16、17、18、19。

该实验采用 2×2 对照：

| 实验变体 | 全局转移 | 结构端点条件下的局部转移 |
|---|---|---|
| `ppr_ppr` | PPR | PPR |
| `ae_ppr_ppr` | AE-PPR | PPR |
| `ppr_ae_ppr` | PPR | AE-PPR |
| `ae_ppr_ae_ppr` | AE-PPR | AE-PPR |

四组实验使用相同的：

- 结构端点来源；
- 重启概率；
- 迭代次数；
- Top-k 数量；
- 攻击预算；
- 特征扰动配置。

唯一受控变化是全局或局部传播中是否加入特征余弦相似度权重。这样可以分别观察全局 AE-PPR、局部 AE-PPR 及二者组合的影响。

## 7. 隐蔽性分析

```bash
bash gbhaa_experiments/launchers/run_stealthiness.sh
```

默认数据集为 Citeseer，攻击预算为 5% 和 10%，随机种子为 15、16、17、18、19。

隐蔽性实验会分别重新运行：

- `edge_only`；
- `feature_only`；
- `parallel`；
- `serial`。

每次攻击保存原始和扰动后的邻接矩阵、节点特征与标签，再进行统一分析。

结构指标：

- 边翻转数量和比例；
- 自环数量；
- 邻接矩阵不对称项数量；
- 度分布 JS 散度；
- 平均和最大节点度变化；
- 连通分量变化；
- 标签同配性变化；
- 全局聚类系数变化。

特征空间指标：

- 被修改节点数量和比例；
- 被修改特征项数量；
- 被修改节点的平均、中位和最小余弦相似度；
- 激活特征集合的 Jaccard 相似度；
- 每个被修改节点平均新增和删除的特征数量；
- 全图平均特征向量变化。

这些指标只反映 **特征空间隐蔽性**。当前攻击产物不保存成对的原始自然语言文本和生成文本，因此不能计算或声称：

- 自然语言语义相似度；
- 文本流畅性；
- BERTScore；
- 人工语言质量。

如果在大图上运行，可通过 `--skip-transitivity` 跳过稀疏三角形乘法，并在结果中说明没有报告全局聚类系数变化。

## 跨随机种子汇总

各实验矩阵将结果追加写入 `runs.jsonl`。汇总程序会：

1. 排除运行失败的记录；
2. 对相同数据集、实验变体、预算和随机种子，只保留最新一次成功结果；
3. 按实验设置分组；
4. 计算均值；
5. 计算样本标准差；
6. 输出 CSV 表格。

使用方式：

```bash
/opt/miniconda3/bin/python -m gbhaa_experiments.aggregate_results \
  gbhaa_experiments/results/gb_ablation/runs.jsonl \
  --output gbhaa_experiments/results/gb_ablation/summary.csv
```

论文正式表格至少应同时给出：

- Clean Accuracy；
- Attacked Accuracy；
- 明确的攻击预算；
- 随机种子数量；
- 均值和标准差；
- 实验硬件信息。

## 低成本验证

```bash
/opt/miniconda3/bin/python -m unittest discover \
  -s gbhaa_experiments/tests -v

/opt/miniconda3/bin/python -m compileall -q gbhaa_experiments

bash -n gbhaa_experiments/launchers/*.sh
```
