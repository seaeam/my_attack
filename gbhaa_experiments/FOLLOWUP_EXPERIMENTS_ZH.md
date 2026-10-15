# AE-PR 后续实验：运行与回填说明

**2026-10-08：补齐运行器的工件身份检查。** 所有套件在标记完成和恢复时，都会核对 `result.json` 与独立 `attack.json` / `evaluation.json` 是否一致，并检查工件中的 seed、数据集、选择器、图状态和全局配置是否对应任务。不同 seed 的旧工件不再因子进程退出码为 0 就被标为完成。启动前即保存实际命令，终端显示 seed、selector、budget；中断时记录异常类型，避免只留下空错误信息。这项修改修复的是完成判定和诊断记录，尚不能解释现有目录中的旧命令和重复工件是如何产生的。

源码指纹会因本次运行器修改而变化。后续定位运行应使用新的 `--output-root`，不要修改旧结果的 seed、指纹或状态来通过检查。先以单个任务检查实际输出，无需立即重跑整套：

```bash
PYTHON_BIN=/Users/shemingcong/anaconda3/envs/smc/bin/python \
  bash gbhaa_experiments/launchers/run_followup.sh selectors \
  --seeds 17 --budgets 0.05 --variants ae_pr \
  --output-root gbhaa_experiments/followup_results_pr_pre_identity_check \
  --execute
```

此命令会实际运行一个攻击并调用配置的 LLM。正确输出应在新目录 `selectors/citeseer/ptb_0.05/seed_17/ae_pr/`；其中 `result.json` 的 seed、command 中的 `--seed` 和独立 `artifacts/attack.json` 的 seed 都应为 17。它是定位任务，不是完整四臂对比，不得与不同源码指纹的旧 cohort 直接合并。

**2026-09-29：factorial 已改为独立的 pre 多种子协议。** 协议名为 `aepr-factorial-v2-pre-stable`，结果默认保存到 `gbhaa_experiments/followup_results_factorial_pre`。保持 degree 全局源、端点局部源、传播公式和翻边前选属性的顺序，只在此新协议接通全局分数降序、同分按节点编号升序的排名。汇总新增两个 global 配对差值，保留两个 local 差值与交互项。旧 factorial 结果不迁移、不改名，新的四臂矩阵需独立运行。

**2026-09-21 当前状态：selectors 主比较已改为 AE-PR / 普通 PR / Degree / Random。** 按作者要求，保留主循环的翻边前属性选择顺序，四臂均声明并使用 `pre` 图。新协议为 `aepr-selectors-v2-pre-pr`，默认结果目录为 `gbhaa_experiments/followup_results_pr_pre`。旧目录中的 PPR、AE-PR、degree 和 random 数字均不能作为新协议结果复用；不能把旧 `ppr` 改名为 `pr`。

`attack.local_graph` 和每轮 `local_graph` 如实写为 `pre`，`requested_local_graph` 单独保存配置要求。ordering 和 uniform_repair 保留旧协议 `aepr-followup-v1-edge-first` 和既有配置；其中请求 `post` 的运行仍与实际顺序不符，汇总会拒绝，当前不能用于证明翻边后传播的作用。旧的四项顺序相关测试仍失败，未通过修改测试掩盖该差异。

9 月 17 日已修复动作和结构计时未记录导致的 `KeyError: structural_seconds`：每次选出端点后记录动作和耗时，即使属性预算已经耗尽也会记录。当时失败发生在 183 次结构扰动完成后的汇总阶段，未保存扰动图或评价结果，不能从日志补成有效运行。内部模型训练与 victim 重训练仍保持现有实现，不能将这些实验称为固定 victim 的 evasion。

**这里没有运行完整攻击实验，也没有调用 LLM。** 完整运行由作者执行。小型合成图测试不产生论文实验数字。

## 1. 已完成的核验

旧结果审计位于 `followup_reports/legacy/audit.md` 和 `audit.json`。共找到 177 条 JSON（含非标准嵌套记录），预期矩阵 210 个单元；37 个单元缺失，16 个已有单元不可直接使用。另有路径与记录身份冲突、重复单元以及准确率/误分类率不一致问题，逐项见报告。

Citeseer 5% 的原 2×2 在配置和指标检查下已有五个完整 seed 区组；10% 有缺失和部分指标冲突。旧记录未保存可核验的源码版本、节点预测和结构轨迹，因此这些检查不等于确认了严格配对或论文数字来源。不能仅补齐缺失文件，就把它们与修复后的程序结果合并。

原全局算子仍为行归一化 `P @ z` 的有限步累积，没有替换成标准 stationary PPR。新主配置明确使用 `degree`。`uniform_repair` 中的 uniform 是退化诊断控制，不是有效 PageRank baseline。全局 degree 不自动证明其排名更有效。

uniform 在该全局算子下理论上产生常数分数。此前用于消除浮点微差的排序方法未被主循环调用；本次为保持旧协议行为，旧 uniform 路径继续使用原来的 NumPy argsort，不宣称已完成此项修复。新 factorial 固定使用 degree 源；稳定排序只处理精确同分，不把微小但非零的分数差合并。

## 2. 新实验协议

四个套件默认使用 seed 15–19。`selectors` 默认保存到 `gbhaa_experiments/followup_results_pr_pre`，`factorial` 保存到 `gbhaa_experiments/followup_results_factorial_pre`；其他套件仍保存到 `gbhaa_experiments/followup_results`。运行器和汇总器均按套件选择默认目录，也可显式指定根目录。

| 套件 | 对照问题 | 当前默认运行数 |
|---|---|---:|
| `selectors` | Citeseer：endpoint AE-PR / 普通 PR / degree / random；5% | 20 |
| `factorial` | Citeseer：global topology/attribute × local endpoint PPR/AE-PR；5% | 20 |
| `ordering` | Citeseer：属性定位使用翻边后图还是翻边前图；5%（post 尚待恢复） | 10 |
| `uniform_repair` | Cora：GB / K-Means / node-level 各自配对 uniform 与 degree；10% | 30 |

当前可运行新的 selectors 和 factorial。显式指定 `--budgets 0.05 0.1` 时，每个套件均为 40 次运行。ordering 的 post 状态问题仍需另行处理。`ordering` 的设计是前后图定位对照，不能替代 serial-versus-parallel 实验。Cora-ML、ACM 主表数字尚未完成逐项来源映射，本脚本不擅自把它们加入统一补跑。

所有新套件使用相同的 seed 分流和模型评价方法；局部排名同分时按节点编号升序。Citeseer 保留原 degree 全局配置 `(alpha_g=.15, N_g=30)`、AE-PR/endpoint PPR 局部 `(.28,60)`；Cora 来源对照保留效率配置 `(.08,180)`、局部 `(.28,60)`。当前每节点最大访问次数为 10。详细参数由 dry-run 命令完整打印。

- 四种 selector 都先排序**全体节点**，再通过相同访问次数/唯一节点可行性规则及每轮候选上限；没有先用 AE-PR 预筛 degree/random 的候选范围。
- 普通 PR 在同一翻边前图上运行 `networkx.pagerank`：忽略属性和端点，以全图均匀分布重启，使用无属性权重的拓扑；去掉自环并保留孤立节点，悬挂概率质量按全图均匀分布重新分配。重启概率 `.28` 对应 NetworkX 的阻尼参数 `alpha=.72`。使用标准概率传播方向，求稳定分布；`max_iter=1000`、`tol=1e-12`，停止阈值为节点数乘以 tol 的 L1 误差，未收敛则报错。PR 不使用局部有限步参数 `text_ppr_iters=60`。求解器设置记录在 `attack.pr_solver`，NetworkX 版本记录在运行环境中。参数语义见 [NetworkX 官方实现](https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html)。
- factorial 中的 `ppr` 仍是双端点重启、去掉余弦边权的局部消融，其定义与结果名称均保留。主比较 AE-PR 对普通 PR 同时改变了属性加权、重启分布和求解方式，因此仅能评估选择器整体效果；属性加权的单独价值需要同端点 PPR 消融，修改后图的价值需要前后图对照。旧 `global_pagerank` 分支不用于新增普通 PR。
- 冻结结构输入特征，每轮独立重置结构 RNG；相同结构设置下重新生成相同的动作轨迹。结果汇总必须核对 `actions_hash` 和最终邻接矩阵，轨迹不相同就拒绝配对。这里是确定性复现并核验，并非从旧实验假设边相同。
- 2×2 中改变全局模块可以改变结构轨迹；只要求同一全局配置的两个局部选项共享结构轨迹。
- clean、edge、feature、combined 的 victim 训练使用同一独立初始化随机流。保留节点 ID、标签与预测，以精确计算 clean accuracy、attacked accuracy、ΔAcc、conditional ASR。
- 生成使用固定 generation seed、temperature=0，服务必须支持 seed 参数。外部服务/硬件仍可能不完全确定，应固定模型版本；不把 seed 参数视为跨服务复现保证。
- 共享的是预算**上限**和约束，不预先声称实际修改条目相等。实际翻边数、修改节点、修改条目和节点访问次数均保存。没有新增强行改满特征条目预算的算法。
- 另存累计特征写入/修改条目、模板缓存命中、节点/簇生成失败，以及实际 API 请求/成功/失败/重试计数。最终净变化与累计写入分开；访问计数不等同于成功修改计数。新严格入口若发生未恢复的部分生成失败，则将该运行排除并保留失败记录。
- Citeseer 延续原 fallback-vocabulary 许可，Cora 延续 partial-vocabulary 许可；这只是保持已有特征映射条件，不能据此宣称真实文本语义或隐蔽性得到验证。

### Factorial 的固定算子与解释范围

- Global：每轮由当前图的未加权度数归一化得到 `v`；`x_0 = .15 v`，随后更新 `x = .85 P x + .15 v` 共 29 次。累计包含初态的 30 个状态，最后按总和归一化。
- Local：当前结构动作两个端点各占一半得到 `v`；`s_0 = v`，更新 `s = .28 v + .72 P.T s` 共 60 次，读取最后状态。`ppr` 与 `ae_pr` 仅切换属性余弦边权，不换成普通 PR。
- 两阶段均行归一化；相似度裁剪到 `[0,1]`；零权重行补自环。结构搜索和 global 使用原始冻结特征；local 使用当前已修改特征，图仍是本轮翻边前状态。
- 新协议把上述规则保存为 `attack.operator_settings`，恢复和汇总均检查。运行时拒绝改变源策略、解除结构特征冻结或偏离固定的 α/迭代数。
- 每轮另存 `global_selected_nodes`、`global_unique_scores`、`global_cutoff_tie_count`（与末位入选分数相同的全图节点数）以及已有分数范围。局部候选仍单独记录在 `selected_nodes`。这些字段位于 `rounds.json`，供排名诊断；当前汇总不回放全局评分，也不把这些字段当作排序正确性的独立证明。
- 此实验是**冻结结构特征条件下的两个阶段加权消融**。只在同一 global 条件内检查局部两臂结构轨迹相同，允许 global 加权改变结构搜索；它不测量局部属性修改反馈到后续结构搜索的完整耦合，也不能替代前后图机制对照。

## 3. 由作者运行

先激活现有包含 PyTorch、DeepRobust、PyG、SciPy 和文本生成依赖的环境。本机可用：

```bash
cd /Users/shemingcong/Downloads/my_attack
export PYTHON_BIN=/Users/shemingcong/anaconda3/envs/smc/bin/python
```

脚本默认连接本地 Ollama 的 OpenAI-compatible 接口 `http://localhost:11434/v1`，模型 `llama3.2:1b-instruct-fp16`。运行前由作者启动已安装服务和模型。可用环境变量 `OPENAI_BASE_URL`、`OLLAMA_MODEL` 覆盖本地配置；远程服务需自行设置 `OPENAI_API_KEY`，不要把密钥写进脚本或结果。代码保留现有远程 GPT 模型选择逻辑，不能仅改 `OLLAMA_MODEL` 就认为远程模型也变更。

先检查计划（不会运行）：

```bash
bash gbhaa_experiments/launchers/run_followup.sh selectors
```

先由作者做一个 seed 的四臂试跑，检查全部完成和配对是否合格：

```bash
bash gbhaa_experiments/launchers/run_followup.sh selectors --seeds 15 --budgets 0.05 --execute
"$PYTHON_BIN" -m gbhaa_experiments.summarize_followup --suites selectors --seeds 15 --budgets 0.05 --output-dir gbhaa_experiments/followup_reports/pr_pre
```

确认试跑正常后运行完整矩阵；同一配置下会跳过已完成的 seed 15 试跑：

```bash
bash gbhaa_experiments/launchers/run_followup.sh selectors --budgets 0.05 0.1 --execute
```

### 3.1 Factorial：global/local 多种子 2×2

使用上面的相同 Python 和 Ollama 环境。先检查计划（不运行）：

```bash
bash gbhaa_experiments/launchers/run_followup.sh factorial --seeds 15 --budgets 0.05
```

先试跑一个 seed、5% 扰动率的四臂，并检查配对产物：

```bash
bash gbhaa_experiments/launchers/run_followup.sh factorial --seeds 15 --budgets 0.05 --execute
"$PYTHON_BIN" -m gbhaa_experiments.summarize_followup --suites factorial --seeds 15 --budgets 0.05 --output-dir gbhaa_experiments/followup_reports/factorial_pre
```

完整运行是 seeds 15–19 × 5%/10% × 四臂，共 40 次；代码、数据和配置未变时，会跳过新目录中已完成且校验通过的试跑：

```bash
bash gbhaa_experiments/launchers/run_followup.sh factorial --seeds 15 16 17 18 19 --budgets 0.05 0.1 --execute
"$PYTHON_BIN" -m gbhaa_experiments.summarize_followup --suites factorial --seeds 15 16 17 18 19 --budgets 0.05 0.1 --output-dir gbhaa_experiments/followup_reports/factorial_pre
"$PYTHON_BIN" -m gbhaa_experiments.plot_followup --summary gbhaa_experiments/followup_reports/factorial_pre/summary.json --output-dir gbhaa_experiments/followup_reports/factorial_pre/figures
```

只计划做 5% 时，运行和汇总都指定 `--budgets 0.05`，共 20 次。不传预算时运行器当前默认只跑 5%，汇总器默认检查 5% 和 10%，因此上述命令均显式传预算。结果自动进入新 factorial 目录，不要指定旧 `followup_results` 来续接历史数字。历史数据需要另做只读审计。

### 3.2 运行状态与恢复

ordering 和 uniform_repair 本次没有调整运行顺序，暂不列入上述可执行矩阵。后续恢复时，`uniform_repair` 的 Cora node-level 非常耗时，可先用 `--variants gb_uniform gb_degree --seeds 15 --budgets 0.05` 试跑，但子集不能作为完整三方法效率结果。各运行串行执行；每个程序使用 1 个计算线程，并启用可检测的确定性限制，不支持的非确定性 kernel 将报错。

程序出错会停止，保留日志和状态。修复外部环境后使用 `--retry-failed`：失败目录先归档，再重跑，不覆盖已完成数字。源代码、输入缓存或配置改变时，使用新的 `--output-root`；不要强行混入旧输出目录。被中断进程可能留下 `.running`，确认原进程已经停止后再移除该锁并重试。

运行时终端实时显示子进程输出和进度条，并同步写入当前运行目录的 `stdout.log`。输出暂时安静时，每 30 秒显示一次 `[RUNNING]` 与已运行时长；每个变体开始/结束显示 `[当前序号/总数]`。中断时已写入的日志会保留。

已经启动的旧运行器不能热加载日志更新，可在另一个终端用 `tail -f logs/对应运行日志.txt` 查看其已有日志。旧运行器的源码检查会在当前变体完成后检测到文件变化并停止后续派发；当前变体的结果仍会保存。继续新版本时使用新的 `--output-root`，保留旧目录，不改写已有结果来源记录。

## 4. 汇总、检查与论文回填

```bash
"$PYTHON_BIN" -m gbhaa_experiments.summarize_followup --suites selectors --budgets 0.05 0.1 --output-dir gbhaa_experiments/followup_reports/pr_pre
"$PYTHON_BIN" -m gbhaa_experiments.plot_followup --summary gbhaa_experiments/followup_reports/pr_pre/summary.json --output-dir gbhaa_experiments/followup_reports/pr_pre/figures
```

上述命令按新 selectors 默认目录读取。只完成 5% 时将 `--budgets` 改为 `0.05`。程序以退出码 2 表示存在缺失、拒绝区组或混合来源，仍会写清单；退出码 0 表示**本次请求的**矩阵通过检查，单 seed 试跑通过并不等于完成五种子实验。审计旧结果仍可使用 `python -m gbhaa_experiments.followup_audit`。

上述汇总写到 `followup_reports/pr_pre`（不传 `--output-dir` 时仍为 `followup_reports/new`）：

- `per_seed.csv`：精确准确率/ASR、实际预算消耗、时间与内存。
- `accuracy_summary.csv`：完整配对区组上的 mean ± sample SD 和 95% t CI。
- `paired_contrasts.csv`：按 seed 配对的准确率差、CI、sign-flip 检验、Holm 校正及 paired dz。
- `summary.json` / `summary.md`：使用了哪些 seed、缺哪些运行、拒绝原因、2×2 interaction。

绘图脚本仅接受检查通过且至少两个完整 seed 的汇总，输出独立 PNG/SVG；不会覆盖论文原图。单 seed 试跑不能生成多种子图。

正的对照差值表示 AE-PR/post/degree-source 的 attacked accuracy 更低。2×2 interaction 定义为 attribute-global 下局部加权的准确率下降增益，减去 topology-global 下相同增益；它不等于无条件机制证明。5 个 seed 的精确双侧 sign-flip 检验分辨率有限，最小 p 为 0.0625，不预设显著性。

新的 factorial 每个预算输出以下 5 个按 seed 配对的差值（准确率百分点）；设 `Y00=topology_ppr`、`Y01=topology_ae_pr`、`Y10=attribute_ppr`、`Y11=attribute_ae_pr`：

| 对比项 | 每个 seed 的计算 |
|---|---|
| `global_gain_with_topology_local` | `Y00 - Y10` |
| `global_gain_with_attribute_local` | `Y01 - Y11` |
| `local_gain_with_topology_global` | `Y00 - Y01` |
| `local_gain_with_attribute_global` | `Y10 - Y11` |
| `interaction_in_accuracy_drop_pp` | `(Y10 - Y11) - (Y00 - Y01)` |

前四项为正表示对应属性加权降低 attacked accuracy；交互项为正表示 global 加权条件下的 local 加权收益更大。先逐 seed 相减，再汇总差值的均值、SD、95% CI 和配对检验。单独汇总 factorial 的两个预算时，Holm 校正覆盖 10 个计划对比。图中均值 ± SD 与 `paired_contrasts.csv` 应共同报告，不能只按最低单元格解释交互。

时间有两个口径：`structural_seconds` 包含内部训练、全局打分、建树、结构定位；`end_to_end_seconds` 包含完整运行和 LLM 等开销。进程峰值内存不包括独立 Ollama 服务。这些数字不能不经口径核对就覆盖旧图的“search time”。

论文只在新结果检查通过后逐项回填：先 selector 表，再 2×2 多种子图，最后确认 Cora 效率结果与旧图来源。主表、摘要优势数字和中文论文暂不自动更新，也不把旧 JSON 的 `misclassification` 当作 `1-combined_accuracy` 使用。

## 投影收益 × 梯度词编辑：四配置、三组匹配快速实验（2026-10-13）

新入口：`gbhaa_experiments/launchers/run_post_editing.sh`。默认 Citeseer、seed 15、固定结构轨迹前 60 次动作，共 **12 次试验**。只运行验证集开发比较，尚不能判断跨种子稳定性。每次仍修改 4 个节点、每节点最多访问 10 次；保留 degree global source、AE-PR alpha=0.28 / 60 次迭代和原训练/预算协议。

| 配置 | 词编辑 | 候选选择 |
|---|---|---|
| `edit_control` | 常见差异词 | 按 seed/round/node 固定随机选择 |
| `edit_gain` | 常见差异词 | 最大化投影后真实候选的固定 surrogate NLL |
| `edit_words` | 梯度建议增删词 | 按 seed/round/node 固定随机选择 |
| `edit_both` | 梯度建议增删词 | 最大化投影后真实候选的固定 surrogate NLL |

每个配置都运行 Endpoint-Post、Endpoint-Pre、Parallel。各组独立地按同一 AE-PR 规则、各自当前属性状态选点；Pre/Post 保留相同结构动作的真实端点，Parallel 使用独立重启点。所有组都使用各自 diffusion 图来计算梯度和候选收益：Post 为翻边后，Pre/Parallel 为翻边前。因此这是**整套属性编辑对图状态的对照**，不再只切换 diffusion 图而固定其余编辑评分图。

四配置每轮都发出 **3 次真实请求，每次请求 1 个模板，max_tokens=50**，最大输出预算合计 150。三个模板共享给本轮四节点；不把单个返回模板复制成三个“候选”。重复返回和重复投影结果照实记录，不额外补采样。此处 `edit_control` 是新的同预算对照，不能当作历史一次请求的 baseline。

词计划在轮初冻结，三个候选槽分别使用 12/24/36 的计划增词上限（梯度方案另有同样的计划删词上限），保留词上限 80、新增条目总上限 70、余弦下限 0.35。梯度方案只建议正梯度的缺失词、负梯度的现有词；实际适配文本排除删除词和计划外新词，再经过真实 vectorizer、预算裁剪、余弦投影。投影可能把删除变成部分降低；记录实际执行增删数。零向量无有效余弦时投影回原向量，按未改变的真实结果记账。仍保留现有 partial vocabulary 的前列对齐规则，未映射部分没有新增词义解释。

按节点 ID 顺序写入。每个候选都在包含此前已接受写入的状态上计算固定模型损失；固定 clean pseudo-labels、仅对非训练节点求损失，候选决策不使用验证/测试真标签。即使三个候选收益全负，也选择一个真实候选并如实报告负收益。保存模型权重、vectorizer、逐轮原始请求/响应、增删词计划、投影候选、实际损失和最终选择，汇总前离线重算核验。

在项目根目录、原 `smc` 环境运行：

```bash
# 预览，不训练、不生成、不创建实验目录
bash gbhaa_experiments/launchers/run_post_editing.sh

# 默认 12 次试验
bash gbhaa_experiments/launchers/run_post_editing.sh --execute

# 跑完后汇总
bash gbhaa_experiments/launchers/run_post_editing.sh summary
```

默认结果目录 `gbhaa_experiments/post_editing_results`，报告目录 `gbhaa_experiments/followup_reports/post_editing`。失败/中断后使用相同参数加 `--execute --retry-failed`，已完整通过审计的试验会跳过。代码或环境指纹变化后请使用新输出目录，不能把旧配置结果混入。

如果从一开始就要两个配对种子（24 次试验），用独立目录：

```bash
bash gbhaa_experiments/launchers/run_post_editing.sh --seeds 15 16 \
  --output-root gbhaa_experiments/post_editing_s15_s16_results --execute
bash gbhaa_experiments/launchers/run_post_editing.sh summary \
  --root gbhaa_experiments/post_editing_s15_s16_results \
  --output-dir gbhaa_experiments/followup_reports/post_editing_s15_s16
```

`paired_comparisons.csv` 给出各配置 Post 对同配置 Pre/Parallel、以及新配置 Post 对 `edit_control` Post 的配对差；正值表示攻击后 ACC 更低。`factorial_effects.csv` 在每组内按 seed 计算收益选择主效应、词策略主效应及交互：设 c/g/w/b 为四配置的 ACC，则收益效应=((c−g)+(w−b))/2，词策略效应=((c−w)+(g−b))/2，交互=g+w−c−b。再报告均值、样本标准差和未校正的配对 t 95% CI。完整性不足时不输出完整效应估计；单种子标准差/CI 未定义。

`per_trial.csv` 另记录不同响应数、不同投影候选数、投影系数、实际计划增删数、累计真实 surrogate 收益及负收益轮数，帮助区分“候选没有多样性”“投影阻断修改”和“surrogate 改善没有迁移到 victim”。这些是开发诊断，不能保证 Post 一定领先。

边界说明：这些是**计划**增删词上限，不是最终向量总减少条目的上限。保留原词最多 80 个，连续特征按 >0.1 提取原词，partial vocabulary 未映射尾部仍使用共同的旧对齐/投影规则；因此实际向量还可能有计划外减少。新增 `editor_unplanned_decreased_entries` / `editor_unplanned_increased_entries` 和逐候选同名诊断会如实计数，收益选择基于包含这些变化的最终投影向量。梯度机制保证指定增删词不会被文本适配逻辑反向覆盖，不宣称每个最终变化条目都由梯度计划指定。
