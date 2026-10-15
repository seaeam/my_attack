# Endpoint-Post 小规模优化：运行与解释

## 1. 目标、默认规模与边界

本方案在已保存的共享结构轨迹上，筛选可能改善 Endpoint-Post 属性定位的配置。
协议标识为 `gbca-post-tuning-v1`。它是开发阶段的小规模筛选，不保证 Post 最好，
也不保证任何配置优于 baseline。本文中的执行命令供作者自行运行；编写说明时未运行真实攻击或 LLM 实验。

默认配置：

- 数据集为 **Citeseer**，种子为 **15**。
- 读取已保存的 **10% 结构预算轨迹**，仅回放其前 **60 个结构动作**。
- `quick` 包含 **10 个 Post 配置 + baseline Endpoint-Pre + baseline Parallel，共 12 个短试验**。
- 所有试验从相同原始特征开始，回放相同的有序结构动作前缀。
- 每个动作固定完成 **4 个不同节点的成功属性写入**，每个节点最多访问 **10 次**。
  60 个动作对应 240 次成功写入机会；这不等于 240 个唯一节点，也不保证每次写入都改变特征。
- 默认只打印计划和子进程命令；必须显式添加 `--execute` 才训练模型、回放攻击并调用生成服务。

`source_budget=0.1` 表示父轨迹的预算。实际短试验预算另存为
`effective_budget = 已回放动作数 / 原始无向边数`。
**前 60 个动作不能写成“完整 10% 攻击”**；增加到 120 个动作同样只是更长前缀。
程序要求父轨迹至少包含所请求的动作数。

排序只使用验证集：**平均 validation ACC 越低越好，ACC 相同时平均 validation NLL 越高越好，
再按 profile ID 稳定排序**。victim 只用训练集标签训练；调优评估保存验证集预测，
不计算 held-out ACC 来选择配置。历史种子 15–19 已有评估记录，seed 16 的复核仍是开发验证，
不能称为未接触过的独立留出测试。

沿用 DeepRobust 保存的数据划分。其 nettack 比例拆分可能因浮点取整而留下未分配节点，
Citeseer 和 Cora 的现有划分各有 1 个这样的节点。入口检查索引有效、集合非空及彼此不重叠，
不要求覆盖所有图节点，也不会把剩余节点补入训练、验证或测试集；仍逐项核验父轨迹的输入哈希。

## 2. 参数到底改变什么

baseline 使用 `alpha=0.28`、60 次扩散更新、零边权下限、零访问惩罚、最后状态读出，
生成上限为 50 tokens。扩散重启分布 `q` 在本轮不同重启节点上均匀分配。

| 配置 ID | 相对 baseline 的变化 | 用途 |
|---|---|---|
| `baseline` | 无变化 | 原有局部扩散和生成配置 |
| `alpha010` | alpha=0.10，120 次更新 | 减少直接重启比例，增加传播 |
| `alpha015` | alpha=0.15，120 次更新 | 较温和地减少重启比例 |
| `alpha050` | alpha=0.50 | 增强重启影响 |
| `floor010` | 边权下限为 0.10 | 让零余弦相似度的现存边也参与传播 |
| `floor025` | 边权下限为 0.25 | 更强的拓扑连接保留 |
| `coverage05` | 访问惩罚指数 gamma=0.5 | 温和降低重复选择节点的排名 |
| `coverage10` | gamma=1 | 更强地降低重复选择节点的排名 |
| `diffused` | 读出为 `max(s-alpha*q, 0)` | 移除最后状态中的直接重启质量 |
| `combined` | alpha=0.15、120 次更新、floor=0.10、gamma=0.5、diffused | 组合候选，需整体解释 |

边权统一采用 `floor + (1-floor) * clipped_cosine`，只作用于当前图中已有的非自环边，
不额外添加图边；行和为零时保留自环兜底。访问惩罚在读出后应用：
`ranking_score = readout / (1+visits)^gamma`。同分时按节点编号升序选择。

这些是明确的**方法变体**。降低 ACC 可能来自新的扩散、覆盖策略或其组合，
不能直接归因为“翻边后图更好”。默认筛选中的 Pre/Parallel 始终使用 baseline；
Post 候选与它们的比较属于探索性比较。要分析图时序的贡献，应再使用同一配置的 matched controls。

三个组的定义保持一致：

| 组 | 用于排名的图 | 重启节点 |
|---|---|---|
| `endpoint_post` | 本轮翻边后图 | 本轮结构动作的两个端点 |
| `endpoint_pre` | 本轮翻边前图 | 相同两个端点 |
| `parallel` | 本轮翻边前图 | 按种子和轮次独立抽取的两个不同节点，允许与端点偶然重合 |

每次 scorer 调用只接收所属组的一张图，不以 Pre/Post 两图差值作为排名输入，也不读取目标标签。
文本分组继续使用各组统一的翻边后图。

### 可选规模

| preset | Post 配置 | 加上默认两个 baseline 对照后的试验数 |
|---|---|---:|
| `tiny` | baseline、floor025、coverage10、combined | 6 |
| `quick` | 上表的 10 个配置 | 12 |
| `extended` | quick 加下列 10 个配置 | 22 |

`extended` 增加 `alpha005`（0.05、240 次）、`alpha075`（0.75）、`floor050`（0.50）、
`coverage025`（0.25）、`combo_coverage`（alpha=0.15、120 次、gamma=1）、
`combo_floor`（floor=0.10、gamma=1），以及以下生成/投影变体：

- `tokens150`：实际生成响应上限从 50 增加到 150 tokens，**扩大了生成预算**。
- `groups2`：文本分组最大节点数从 4 改为 2，每个结构动作仍固定完成 4 次节点写入。
- `added35`：最大新增词数从 70 改为 35。
- `cdl18`：投影的 CDL top-k 从 36 改为 18。

后四项改变了生成或投影条件，不能作为仅隔离 Post 翻边机制的证据。
实际 `max_tokens` 会在生成前设置，并进入完整请求缓存键和元数据。

## 3. 准备环境与父轨迹

在现有 `smc` 环境中运行：

```bash
cd /Users/shemingcong/Downloads/my_attack
export PYTHON_BIN=/Users/shemingcong/anaconda3/envs/smc/bin/python
```

默认父结果目录是 `gbhaa_experiments/followup_results_action_state`。
默认 Citeseer / 10% / seed 15 的父记录应位于：

```text
gbhaa_experiments/followup_results_action_state/citeseer/ptb_0.1/seed_15/structure/result.json
```

同目录下的结构图、动作和轮次产物也必须保留。执行时验证父协议、已保存的指纹、输入、动作哈希
及图回放一致性。缺少产物会报出具体路径；程序不会自动重新规划昂贵的结构攻击，也不会借用其他种子的动作。
如果父轨迹位于其他目录，所有相关命令使用相同的 `--parent-root <目录>`。
seed 16 的后续复核需要对应的 `seed_16/structure` 产物。

真实执行需要可用的生成服务。默认连接本地 Ollama 的
`http://localhost:11434/v1`，模型为 `llama3.2:1b-instruct-fp16`。
作者应先准备已有服务和模型，再设置：

```bash
export OPENAI_API_KEY=ollama
export OLLAMA_MODEL=llama3.2:1b-instruct-fp16
```

这里的 `ollama` 是本地兼容接口的占位值。使用远程接口时自行设置真实的 `OPENAI_API_KEY`，
并通过 `--api-base-url` 或 `OPENAI_BASE_URL` 指定服务；不要把密钥写入结果或脚本。
`OLLAMA_MODEL` 用于现有本地 Ollama 路径，不能据此声称远程模型选择也改变。

## 4. 先看计划，再运行和汇总

### 默认 quick：12 个试验，仅预览

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_post_tuning --python "$PYTHON_BIN"
```

预览应显示 `PLAN ONLY: 12 short trials, 60 actions each`。此时不训练、不调用生成服务，
也不会生成真实结果 `plan.json`。可单独查看全部配置说明：

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_post_tuning --list-profiles
```

也可以使用已提供的脚本 `bash gbhaa_experiments/launchers/run_post_tuning.sh`。
脚本读取 `PYTHON_BIN`，其余参数与上述 runner 相同；加 `--execute` 即运行默认 12 个试验。
未指定 `PYTHON_BIN` 时，脚本优先使用当前 `CONDA_PREFIX/bin/python`，没有激活 Conda 环境才使用 PATH 中的 `python`。
实际执行会在保存计划前，用指定解释器检查 `TextAttackGenerator` 及其依赖能否导入。

### 确定运行默认筛选时

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_post_tuning \
  --python "$PYTHON_BIN" \
  --datasets citeseer --seeds 15 --source-budget 0.1 --max-actions 60 \
  --preset quick --output-root gbhaa_experiments/post_tuning_results \
  --execute
```

runner 串行执行，失败即停止并保留日志。不同配置的生成缓存放在本次结果根目录下，
按当前源码、输入和环境指纹隔离；只有完整请求与生成设置一致才复用回复。

### 汇总与推荐

```bash
"$PYTHON_BIN" -m gbhaa_experiments.summarize_post_tuning \
  --root gbhaa_experiments/post_tuning_results \
  --output-dir gbhaa_experiments/followup_reports/post_tuning
```

汇总的参数是 `--root` 和 **`--output-dir`**；runner 保存结果使用的是 `--output-root`。
汇总读取 `plan.json` 中完整的请求矩阵，不从现有成功目录反推应有试验：

- `per_trial.csv`：每个试验的状态、验证 ACC/NLL、实际预算、实际变化节点/条目、L1/L2/Linf、
  成功写入、耗时、内存、覆盖率和端点占比。
- `ranking.csv`：Post 配置的验证集排名、有效种子数和样本标准差；单种子样本标准差未定义。
- `summary.json` / `summary.md`：排名、缺失或拒收原因，以及相对 baseline Post 和可用 Pre/Parallel 的差值。
- `recommendation.json`：可供后续 matched 复核读取的开发候选。

只有**全部请求试验、全部请求种子都通过校验且匹配**时才给出推荐。
存在缺失、失败、拒收或不匹配记录时，部分排名可以展示，但不会推荐；汇总返回码为 2。
完整汇总返回码为 0。不能只删除失败行、缩小原计划或挑选成功种子来获得推荐。

### 可选 tiny / extended：先用独立目录预览

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_post_tuning \
  --python "$PYTHON_BIN" --preset tiny \
  --output-root gbhaa_experiments/post_tuning_results_tiny

"$PYTHON_BIN" -m gbhaa_experiments.run_post_tuning \
  --python "$PYTHON_BIN" --preset extended \
  --output-root gbhaa_experiments/post_tuning_results_extended
```

分别为 6 和 22 个试验。确定使用该规模后，在对应命令末尾添加 `--execute`；
汇总时让 `--root` 指向对应结果目录，并为报告使用独立的 `--output-dir`。
`--profiles <ID...>` 可以替代 preset 的配置列表；baseline Post 始终保留。

## 5. 可选：用 seed 16 做匹配配置复核

先完成上面的筛选和汇总，再把生成的推荐文件传给 runner。
`--recommendation` 会选取获推荐的配置并自动启用 matched controls，不能同时使用 `--profiles`。
它要求复核种子与筛选种子不同，并验证推荐中的配置仍与当前配置定义一致。

先预览：

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_post_tuning \
  --python "$PYTHON_BIN" \
  --recommendation gbhaa_experiments/followup_reports/post_tuning/recommendation.json \
  --datasets citeseer --seeds 16 --source-budget 0.1 --max-actions 60 \
  --output-root gbhaa_experiments/post_tuning_matched_seed16
```

该计划包含获推荐配置的 **Post、Pre、Parallel 三组**，再保留 **baseline Post**，通常共 **4 个试验**。
若获推荐配置就是 baseline，则三组已包含 baseline Post，共 **3 个试验**。
没有完整推荐文件或对应父轨迹时不能执行该流程。

决定执行时，保持上述参数并添加 `--execute`：

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_post_tuning \
  --python "$PYTHON_BIN" \
  --recommendation gbhaa_experiments/followup_reports/post_tuning/recommendation.json \
  --datasets citeseer --seeds 16 --source-budget 0.1 --max-actions 60 \
  --output-root gbhaa_experiments/post_tuning_matched_seed16 --execute

"$PYTHON_BIN" -m gbhaa_experiments.summarize_post_tuning \
  --root gbhaa_experiments/post_tuning_matched_seed16 \
  --output-dir gbhaa_experiments/followup_reports/post_tuning_matched_seed16
```

也可以手动指定如 `--profiles combined --matched-controls`，此时配置由作者选择，
不能写成程序已选出的获胜者。匹配复核能减少配置不一致的干扰，但一个种子、一个短前缀仍不足以建立稳定结论。

若需要可选的 120 动作复核，将上述 runner 参数改为 `--max-actions 120`，并同时换用新目录
`--output-root gbhaa_experiments/post_tuning_matched_seed16_steps120`；报告也使用对应新目录。
父轨迹必须至少有 120 个动作。这个结果应标为“120 动作前缀”，不可与 60 动作结果混写为同一预算。

## 6. 重试、复用与结果隔离

对相同配置、源码、输入、环境和父轨迹，重新运行原命令会先验证再跳过已完成试验。
失败或中断试验需要显式 `--retry-failed`，例如：

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_post_tuning \
  --python "$PYTHON_BIN" \
  --datasets citeseer --seeds 15 --source-budget 0.1 --max-actions 60 \
  --preset quick --output-root gbhaa_experiments/post_tuning_results \
  --execute --retry-failed
```

失败/未完成目录会先改名归档，再从该试验开头重跑；这不是逐轮续训。
`--retry-failed` 不会绕过配置或指纹不一致检查。
变更 preset、profile、种子、动作数、源码、数据、模型或环境后，使用新的 `--output-root`，
不要覆盖旧 `plan.json` 或强行修改指纹。报告目录也应分别保存。

如果遇到 `Saved plan differs`，错误会列出具体变动字段，以及旧计划和当前的 Python 路径。
例如，先用 base Python 创建了失败计划，之后切换到 smc，其依赖版本不同，不能直接续用旧计划。
此时保留旧目录并指定新的结果目录；`--retry-failed` 仅用于相同计划下的失败重试。
可以显式固定解释器，避免终端中的别名或 PATH 选错环境：

```bash
PYTHON_BIN=/Users/shemingcong/anaconda3/envs/smc/bin/python \
  bash gbhaa_experiments/launchers/run_post_tuning.sh --execute
```

当前 `source_manifest()` 会纳入项目根目录以及 `gbhaa_experiments/`、`tools/` 中的 Python 源码。
因此新增的调优 `.py` 文件也可能使旧 runner 的继续运行检查发现源码集合已改变。
这不表示旧结果被修改；旧结果仍应保留，后续执行遵循其指纹检查，并使用新的结果根目录。

调优默认结果位于 `gbhaa_experiments/post_tuning_results`，默认报告位于
`gbhaa_experiments/followup_reports/post_tuning`。父结构产物只读使用。
runner 禁止把输出放进历史结果目录或父目录树。本流程不替换原有 60 次三组攻击结果，
不改写论文，不把开发筛选重新命名为正式独立测试，也不预设 Endpoint-Post 必须获胜。

## 7. 实现验证记录

2026-10-12：新增流程及相关回归共 156 项离线测试通过，覆盖参数解析与生效、原始选择结果一致性、
真实回放入口的合成产物校验、验证集评估、缓存、失败归档与续跑。脚本语法及 6/12/22 个试验的预览通过。
这些是代码验证，不代表真实数据集上已经获得优化收益。

另行运行旧 `tests/test_followup_protocol.py`：13 项中有 4 项失败，原因是这些旧测试要求
`Heirattack` 原主循环改为翻边后选择，并依赖当前不存在的 `prepare_attribute_adjacency`。
按作者此前要求，本次保留原主循环，因此没有为通过这些旧测试而修改旧代码或测试。
新增入口单独实现所需时序；原 `heir.py`、`meta.py`、原 action-state 代码和论文文件的哈希保持不变。
