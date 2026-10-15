# Post 下一轮：代理一致性与计划特征保留

## 为什么做这一轮

上一轮 Citeseer / seed 15 / 60 动作：组合 Post 对固定代理的 ACC
降至 60.6635%，最终重训练 victim 的 ACC 却为 69.1943%，弱于对照的
66.8246%。编辑代理无 ReLU，victim 有 ReLU且重训练，存在迁移差距；
不能把所有差距归因于某一项。

只读回放还发现组合 Post 的 6,708 次计划外降低中，6,598 次在词表
之外。词表只覆盖前 500 个特征，实际特征宽度 3,703。旧替换方式先
补零尾部，再进行余弦投影，尾部最终 L1 降低为 4,560.731。
这些是旧轨迹上的统计，不能用于宣称新方案有效。

优先级：

1. 用独立、与 victim 架构及拟合规则一致的干净代理评分和选词。
2. 用明确的词特征增删代替全向量替换，保留非计划修改的坐标。
3. 以后再考虑收益/L1、边界目标、或包含训练过程的梯度。收益/L1在
   保存的组合 Post 状态上只改变 23/240 次 argmax；本轮不叠加该因素。

## 五配置，三组全部匹配

| 配置 | 编辑代理 | 特征更新 | 用途 |
|---|---|---|---|
| edit_control | 原固定代理 | 原替换；频率词、随机候选 | 上轮最佳 Post 参考 |
| edit_both | 原固定代理 | 原替换；梯度词、投影 NLL 选候选 | 新 2×2 起点 |
| transfer_relu | 独立训练的 ReLU 代理 | 与 edit_both 相同 | 仅代理策略改变 |
| transfer_residual | 原固定代理 | 仅计划增删坐标 | 仅特征更新策略改变 |
| transfer_both | 独立训练的 ReLU 代理 | 仅计划增删坐标 | 组合 |

新代理固定使用干净图/属性，CPU、200 epoch、Adam、lr=.01、weight decay=.0005，
hidden/dropout/bias 与原模型配置一致，ReLU开启。种子为独立的
`seed_stream(seed, "editor-transfer")`，不使用最终 victim 的随机初始化。
仅传入训练节点标签；不看验证标签，也不用验证集早停。结构代理、原始
固定伪标签和 all-nontraining 攻击目标集均保持不变。这是代理策略的
整体消融，不是纯 ReLU 开关消融，也没有模拟 victim 后续的重训练。
本轮脚本使用默认 hidden=16、dropout=.5。若直接调用底层入口修改 victim
参数，导致其架构与结构代理不同，程序会在拟合新代理前明确报错。

保留方案从**当前**特征出发，只把实际 vectorizer 支持的计划增词写入、
计划删词置零，再按原新增词上限70、余弦下限.35投影。其他坐标精确保留，
包括词表外特征和以前轮次留下的小数值。它是部分词表下的特征补丁，
不等同于整段文本替换，也没有修复“词表与原数据列语义映射尚未核实”的限制。
三个候选仍由各自词计划约束，经真实向量化后评分；在二值词表中，
强制执行词计划会使保留方案的最终向量主要由计划决定，生成措辞未必改变向量。

每个配置都运行 Post/Pre/Parallel。同一结构轨迹、选点规则、每轮4次
写入和3次逻辑生成请求、候选数3、词上限12/24/36、种子、训练协议一致。
Post 的扩散/梯度/候选评分都用翻边后的图；Pre使用翻边前图及相同端点；
Parallel使用翻边前图及独立重启节点。比较的是整套局部编辑图状态。
随着特征演化，实际节点和扰动幅度可以不同，必须在报告中同时查看。
无可行词修改时可成功写入原值，不能额外抽样补充预算或隐藏这种结果。

## 运行

在项目目录、smc 环境中：

```bash
# 预览：15项，不训练、不调用LLM、不创建结果目录
bash gbhaa_experiments/launchers/run_post_transfer.sh

# 正式快速比较：Citeseer，seed15，60次结构动作
bash gbhaa_experiments/launchers/run_post_transfer.sh --execute

# 完成后审计并汇总
bash gbhaa_experiments/launchers/run_post_transfer.sh summary
```

默认新目录 `gbhaa_experiments/post_transfer_results`，报告在
`gbhaa_experiments/followup_reports/post_transfer`。继承现有 Ollama/API
环境和结构轨迹输入。新目录不混入旧实验；执行时会新增代理训练（仅ReLU配置），
其余成本仍主要来自原有回放、生成和最终 victim 训练。

中断且已确认没有同一试验进程运行时，用 `--execute --retry-failed`。
它会归档失败项后重试。代码、环境或计划改变时使用新 `--output-root`，
不要删除旧结果来规避指纹检查。

若希望直接比较两个种子（共30项），另开目录：

```bash
bash gbhaa_experiments/launchers/run_post_transfer.sh --execute \
  --seeds 15 16 --output-root gbhaa_experiments/post_transfer_seeds15_16
bash gbhaa_experiments/launchers/run_post_transfer.sh summary \
  --root gbhaa_experiments/post_transfer_seeds15_16 \
  --output-dir gbhaa_experiments/followup_reports/post_transfer_seeds15_16
```

## 判断标准

- ACC越低攻击越强；先比较每个配置内 Post−Pre、Post−Parallel 的配对
  结果，再比较新 Post 与 edit_control / edit_both Post。汇总差值采用
  对照ACC−Post ACC，正值代表Post更强。
- `transfer_effects.csv` 是以 edit_both 为起点的代理/保留 2×2 效应，
  先按 seed/arm 作差，再计算均值、样本SD和95% t区间。
- `per_trial.csv` 报告实际幅度/节点覆盖/重复率、真实变化写入数、词表外
  降低次数、候选选择次数。候选NLL由不同模型评价，不能跨代理直接比数值大小。
- 仅 seed15 是筛选证据，不能称“稳定领先”。两个种子也只是复查；15–19
  都已有开发历史，不是未见过的独立测试。只有完整匹配区组才支持比较。
- 程序不保证Post获胜；若新机制只改善固定代理而不改善重训练victim，
  应继续研究包含训练过程的目标，而不是只提高原代理损失。

旧37个配置保持定义不变；`heir.py`、原回放顺序和论文正文未修改。
