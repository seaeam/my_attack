# Pre-flip / Post-flip 配对消融运行说明

## 1. 实验检验什么

该方案适合检验：在相同结构动作下，修改后的传播图是否能改善属性定位。
它不能预先保证 Post 更好，也不能仅凭一次对照证明 Post 在所有条件下“必不可少”。

主比较是 **Endpoint-Pre 与 Endpoint-Post**。Parallel 是端点耦合的辅助对照。
为避免把选择器、预算等其他因素混进来，本实验固定如下定义：

| 实验组 | 局部扩散图 | 重启节点 | 排名方法 |
|---|---|---|---|
| Parallel | 修改前的图 | 按种子和轮次独立抽取两个不同节点 | 属性加权 AE-PR |
| Endpoint-Pre | 修改前的图 | 本轮结构动作的两个端点 | 属性加权 AE-PR |
| Endpoint-Post | 执行翻边后的图 | 同样的两个端点 | 属性加权 AE-PR |

Parallel 的重启节点从全部节点独立抽取，允许偶然与端点重合；刻意排除端点反而会使其依赖结构动作。
这个 Parallel 是本次明确约定的独立选择对照，不直接沿用旧表中的 Parallel 数字。
三组都不是普通 PageRank 对照；它们使用相同的局部 AE-PR 算子。

每个 dataset / budget / seed 区组先生成一份共享的结构轨迹，再由三组回放。
结构规划使用 degree 全局源、属性加权全局传播、冻结的原始结构特征。
因此这是固定结构动作下的机制消融，不是允许结构路径各自变化的完整端到端方法比较。

三组使用同样的初始化、合法候选全集、访问次数上限、生成和投影代码及 GCN victim 训练协议。
局部扩散的重启系数为 0.28、迭代 60 次、读出最后状态；并列分数按节点编号排序。
用于文本分组的图在三组中统一为翻边后图；训练分组模型时统一使用原始特征和当轮修改前图。

**属性预算：每轮恰好 4 个不同节点写入，每节点最多访问 10 次。**
结构预算为 B 条边时，各组都有 4B 次成功节点更新机会；不设额外的总唯一节点数上限。
这不等于最终修改 4B 个唯一节点或完全相同的特征条目数。
成功生成但没有改变特征的写入仍计入预算，避免按实验结果筛除样本；部分写入或生成失败则标为失败。
汇总另外报告实际修改节点数、条目数和 L1/L2/Linf 修改幅度。

三组的特征可随着各自更新逐渐分化，所以最终 ACC 比较的是固定结构轨迹下的累积策略效果。
每轮还在同一组的同一特征快照上计算 Pre/Post 探针，记录两者的排名分数 L1 差异、选中集合 Jaccard
和端点属性相似度，保存在 `artifacts/rounds.json`。这有助于区分“翻边没有改变传播结果”和
“传播结果改变了但没有改善攻击效果”。

## 2. 默认规模与环境

- 数据集：Citeseer、Cora。
- 结构预算：0.05、0.10。
- 配对种子：15–19。
- 共 **20 份结构轨迹 + 60 次三组攻击**。结构规划不调用 LLM，也不单独评估 victim。
- 默认本地 Ollama：`http://localhost:11434/v1`，模型 `llama3.2:1b-instruct-fp16`。
- 默认结果目录：`gbhaa_experiments/followup_results_action_state`。

沿用当前项目的数据、真实 BoW 词表和依赖环境。脚本允许已知的短词表按现有前导列投影规则处理，
不允许用人工占位词表代替真实词表；实际词表模式会记录在结果中。
先启动本地模型服务，并确认指定模型已经安装。运行期间保持源码、数据、词表和模型版本不变。

耗时应由一个区组的实测时间估算；6–24 小时 / dataset 只是原建议的估计，并非脚本保证。
共享请求缓存以完整提示词、模型、生成 seed、temperature 和 max_tokens 为键；请求相同才复用回复。
三组提示词可能不同，因而不能假定三组全部命中缓存。

## 3. 运行命令

进入项目，使用现有 Python 环境：

```bash
cd /Users/shemingcong/Downloads/my_attack
export PYTHON_BIN=/Users/shemingcong/anaconda3/envs/smc/bin/python
```

### 只预览任务（默认行为，不训练、不调用 LLM）

```bash
bash gbhaa_experiments/launchers/run_action_state.sh
```

### 先跑一个完整区组，检查环境和耗时

```bash
bash gbhaa_experiments/launchers/run_action_state.sh \
  --datasets citeseer --seeds 15 --budgets 0.05 --execute
```

单区组包含一份共享结构轨迹及三组攻击。它用于检查流程，不能用于多种子结论。

### 跑完默认实验

```bash
bash gbhaa_experiments/launchers/run_action_state.sh --execute
```

如果先前区组的配置、源码、输入和环境一致，脚本会校验文件后跳过已完成结果。
运行串行执行；同一结果目录同时只允许一个 runner。
失败时停止并打印具体 `stdout.log` 路径。

### 失败或中断后重试

```bash
bash gbhaa_experiments/launchers/run_action_state.sh --execute --retry-failed
```

这会归档失败 / 未完成目录，再从该阶段开头重跑，保留已完成阶段和有效 LLM 请求缓存。
它不是逐轮断点续训。源码、数据、环境或模型配置变化后不能混用旧结果；请指定新目录，例如：

```bash
bash gbhaa_experiments/launchers/run_action_state.sh \
  --output-root gbhaa_experiments/followup_results_action_state_v2 --execute
```

### 可选：增加 Pubmed

```bash
bash gbhaa_experiments/launchers/run_action_state.sh --datasets pubmed --execute
```

Pubmed 更大，建议先用 `--datasets pubmed --seeds 15 --budgets 0.05 --execute` 检查耗时和内存。
若仅希望一个预算，可以在运行和汇总时都指定 `--budgets 0.05`。

## 4. 汇总和解释

默认两数据集、两预算、五种子：

```bash
"$PYTHON_BIN" -m gbhaa_experiments.summarize_action_state
```

只查看试运行区组：

```bash
"$PYTHON_BIN" -m gbhaa_experiments.summarize_action_state \
  --datasets citeseer --seeds 15 --budgets 0.05
```

包含 Pubmed 时增加 `--datasets citeseer cora pubmed`。
使用其他结果目录时增加 `--root <结果目录>`，需要时用 `--output-dir <报告目录>` 单独保存报告。

默认报告在 `gbhaa_experiments/followup_reports/action_state/`：

- `summary.md` / `summary.json`：总体结果及缺失、失败、校验异常清单。
- `per_seed.csv`：各种子 ACC、真实扰动量和资源开销。
- `accuracy_summary.csv`：各组 ACC 均值、样本标准差和均值的 95% CI。
- `paired_differences.csv`：每个种子的配对 ACC 差值。
- `paired_contrasts.csv`：配对差值均值、样本标准差、95% CI，以及辅助检验。

只有结构轨迹及三组攻击都完整且匹配的种子区组才进入汇总。
脚本读取实际图、逐轮特征写入和预测文件核验，不能仅凭存在 `result.json` 判为完成。
汇总返回码 0 表示请求的区组全部完整；返回码 2 表示存在缺失或异常，报告仍会写出。
单种子试运行可以返回 0，但仍明确标为 preliminary，标准差和 CI 留空。

主差值定义为 **ACC(Pre) − ACC(Post)**，单位为百分点：

- 正值表示 Post 的被攻击准确率更低，攻击效果更强。
- 若 95% CI 跨过 0，当前种子数量不足以从该区间排除无差异。
- CI 为基于配对种子的 t 区间，五种子时是小样本近似；不能把节点数量当成独立重复次数。

Parallel−Post 与 Parallel−Pre 为辅助比较。
报告额外给出主比较的双侧精确符号翻转检验，并在请求的数据集 / 预算组合间进行 Holm 校正。
五个种子时，双侧精确检验可达到的最小 p 值为 0.0625，因此不要把它作为五种子必须“显著”的门槛；
应先报告效应大小、配对差异和区间，必要时再预先规划更多种子。

## 5. 与现有实验的关系

此功能是独立入口，使用单独的协议和输出目录。
旧主循环的属性选择 / 翻边顺序没有改变；本实验结果不回填或重命名为旧实验结果。
论文正文没有修改。该脚本不会自动启动正式实验，只有显式 `--execute` 才运行。
