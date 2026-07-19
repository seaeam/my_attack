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
└── launchers/                       # 每项实验对应的启动脚本
```

实验结果统一写入 `gbhaa_experiments/results/`。每次实际运行会保存：

- 完整的标准输出和错误日志；
- 已隐藏 API Key 的实验命令；
- 整条攻击命令的墙钟时间，包括阻塞等待外部 LLM 响应的时间；
- 系统支持时由 `/usr/bin/time` 记录攻击命令进程的峰值常驻内存（不包括独立 Ollama 进程及其 GPU 显存）；
- 操作系统、Python 版本、CPU 数量和可见 CUDA 设备；
- 系统存在 `nvidia-smi` 时记录 GPU 型号和驱动版本；
- 干净图、结构攻击、特征攻击和联合攻击准确率；
- 数据集、攻击预算、随机种子和实验变体信息。

## 执行策略

所有启动脚本现在默认直接执行正式实验。完整实验矩阵包含大量 GNN 训练和 LLM 请求，请在运行前确认当前 Python 环境、数据和模型服务已经准备好：

```bash
# 在仓库根目录执行
bash gbhaa_experiments/launchers/run_gb_ablation.sh
```

启动脚本默认使用当前环境中 `command -v python` 解析到的解释器。如果需要覆盖，可通过 `PYTHON_BIN` 指定：

```bash
PYTHON_BIN=/path/to/python \
  bash gbhaa_experiments/launchers/run_efficiency.sh
```

如需只预览命令而不执行实验，可显式设置 `EXECUTE=0`：

```bash
EXECUTE=0 bash gbhaa_experiments/launchers/run_efficiency.sh
```

只有实际读取 `DATASETS` 的 `run_gb_ablation.sh`、`run_hybrid_ablation.sh`、`run_llm_ablation.sh`、`run_ae_ppr_comparison.sh` 和 `run_stealthiness.sh` 可以用该变量覆盖默认数据集，例如：

```bash
DATASETS="citeseer" \
  bash gbhaa_experiments/launchers/run_hybrid_ablation.sh
```

`run_efficiency.sh` 固定串行 Citeseer 和 Cora，不读取 `DATASETS`，因此不能用该变量改变效率矩阵。

需要外部模型的实验默认连接本地 Ollama 兼容接口。可通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OLLAMA_MODEL` 切换服务或模型。实验元数据不会保存明文 API Key。

## 特征词表与报告边界

效率实验固定为 Citeseer 和 Cora，分别使用现有 `bow_cache/citeseer.pkl` 和 `bow_cache/cora.pkl` 中的 500 个可读词项。该启动器显式传入 `--allow_partial_vocabulary`；当前代码只保留缓存顺序并按位置映射到特征矩阵最前面的 500 列（Citeseer 共 3,703 维，Cora 共 1,433 维），没有验证这些词项与数据集原始特征名在语义上同一。不显式选择部分缓存词表或回退词表时，词表与特征维数不匹配仍会严格报错。

效率实验的属性写回路径是：LLM 生成文本先仅在这 500 个缓存词上向量化，再向 3,703/1,433 维特征宽度补零，然后依次应用新增词上限和余弦相似度投影。因此，补零后的原始生成向量在后续列上是 0；若未达相似度下限，与原始完整特征做线性插值时可按比例恢复这些后续列。

效率启动器不使用 `feature_i`，也不传入 `--allow_fallback_vocabulary`。它可以表述为遵循历史脚本的可读缓存词表路径，但因为上述位置映射未验证语义同一性，仍不能声称自然语言语义有效或语义保持。其他仍显式传入回退词表参数的启动器则是离散特征空间消融：`feature_i` 仅表示特征列，不得据此声称自然语言语义攻击、语义保持或文本流畅性。

## 1. 端到端效率实验

```bash
bash gbhaa_experiments/launchers/run_efficiency.sh
```

这是尚待执行的 90 个完整攻击运行协议：数据集固定为 Citeseer 和 Cora，攻击预算为 1%、5% 和 10%，随机种子为 15、16、17、18、19，且两个数据集都运行 `gb`、`kmeans` 和 `node_level`。每个子任务都通过 `meta.py` 执行结构攻击与外部 LLM 属性攻击。

实验变体：

- `gb`：Granular-Ball 层次结构搜索；
- `kmeans`：使用相同聚类数量的 K-Means 层次搜索；
- `node_level`：设置 `--level 1`，将候选细化为单节点，作为无粗化参考。

主指标为 Combined Accuracy，同时记录：

- Clean、Edge、Feature 和 Combined Accuracy，以及各攻击相对 Clean Accuracy 的 drop；
- `LLM Calls` 与 `Cache Hits`：前者是当前代码记录的 cluster-level 模板生成缓存未命中/生成调用计数，后者是对应模板缓存命中数；API 传输重试不会单独增加 `LLM Calls`，因此它不是 HTTP 请求数或成本计数，实际请求数应使用 Ollama/provider telemetry；
- 实际完成的结构扰动数；
- 整条攻击命令的墙钟时间（包括阻塞的 LLM 延迟）；
- 仅在 `/usr/bin/time` 存在且输出可解析时，记录攻击命令进程的峰值 RSS；该值不包括独立 Ollama 服务进程及其 GPU 显存。

效率的主比较是 `gb` 与 `node_level`；`gb` 与 `kmeans` 的对照只用于隔离粗化分组类型的影响。效率启动器只复制根目录 [`run_citeseer_gb.sh`](../run_citeseer_gb.sh) 和 [`run_cora.sh`](../run_cora.sh) 中列出的数据集专用参数值：Citeseer 使用 level/miter/lr = 4/60/0.05，Cora 使用 2/450/0.012；PPR、属性预算和文本约束的完整取值见[完整协议](./EXPERIMENT_SUPPLEMENT_PLAN_ZH.md#23-固定参数)。效率入口始终统一为 `meta.py`；`run_cora.sh` 当前的实际执行行可能是 `meta_gsage.py`，但本实验不复制该入口。上述均为计划协议，不代表已获得的正式结果。

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
python -m gbhaa_experiments.aggregate_results \
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

正式矩阵计划为每组 5 个种子，但聚合会先排除失败记录，并对每个数据集/变体/预算/种子只保留最新的成功记录，所以实际 `n` 可能小于 5。论文的完整正式行必须确认 `n=5`，并逐组披露 `n`；不得将失败被排除后的行笼统表述为“五个种子的统计”。

## 低成本验证

```bash
python -m compileall -q gbhaa_experiments

bash -n gbhaa_experiments/launchers/*.sh

EXECUTE=0 bash gbhaa_experiments/launchers/run_efficiency.sh
```
