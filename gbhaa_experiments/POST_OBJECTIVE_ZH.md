# ReLU 基础上的目标函数快速比较

## 这轮解决什么

上一轮仅换 ReLU 代理的 Post 在 Citeseer / seed 15 上比 Pre 低 0.9479
个百分点、比 Parallel 低 0.4739 个百分点，但只有一个种子，不能称为稳定优势。
它也没有超过更早的 `edit_control`。本轮按要求以 `transfer_relu` 为起点。

对已保存的 240 次 Post 写入作同状态候选回放，发现正向伪标签 NLL 增量中
62.74% 来自已经偏离原伪标签的节点。把目标改成跨过分类边界后不再奖励，
会改变 92/240 次候选选择：三档词数的选中次数从 3/1/236 变为 85/10/145。
这没有重新生成或训练，也没有沿新选择继续演化，只用于发现优化方向，
不是新方案的 ACC 结果。

## 四个配置，每个都运行 Post、Pre、Parallel

| 配置 | 评分目标 | 目标节点权重 |
|---|---|---|
| `transfer_relu` | 原伪标签 NLL | 均匀；原配置不变 |
| `objective_margin` | 分类边界截断目标 | 均匀 |
| `objective_local` | 原伪标签 NLL | 50% 均匀 + 50% 本组局部扩散 |
| `objective_both` | 分类边界截断目标 | 50% 均匀 + 50% 本组局部扩散 |

边界目标为 `-relu(logp[固定伪标签] - max_other_logp)`，取加权均值并最大化。
达到分类边界后，该目标项的梯度为零；不会继续奖励把同一个节点推得更远。
它上界为零，仍使用原结构代理在干净数据上的固定伪标签，不用非训练节点真标签。

局部权重在原来的全部非训练目标节点上归一化，再与均匀权重各取一半。
所有目标仍有正权重；不改变候选资格，也不先用 Post 筛候选池。
局部扩散质量小于等于 `1e-12` 时回退为均匀权重并记录。
每轮开始时固定权重，选词梯度和投影后真实候选评分共用同一目标。
当前轮内顺序写入引起的特征变化不会重新改变这轮的权重。

## 三组如何匹配

- Post：翻边后图、实际边的两个端点。
- Pre：翻边前图、完全相同的两个端点。
- Parallel：翻边前图、原协议独立采样的重启节点。

各组的选点、局部权重、梯度、候选评分均使用自己的图和重启点。
每组沿自己的属性状态继续运行。对比检验的是整套局部编辑对图状态的依赖，
不是仅改变扩散时间、同时固定后续评分图的单因素实验。

同种子共用相同结构动作、干净 ReLU 代理参数、固定伪标签、目标集、训练划分
和 victim 训练协议。每轮仍是 4 次写入、3 次真实生成请求、3 个候选、
词数上限 12/24/36、输出上限 50 tokens，投影规则、访问上限、选点规则不变。
保留原完整替换方式，不启用上一轮表现较差的 `editor_residual`。
记录实际修改节点数/条目数/L1、无效或未变化写入、覆盖和重复率。

新审计从结构动作和组别重建图及重启点，逐轮重算权重、词计划、投影候选和
最终选择，验证 objective 类型、权重哈希、回退情况等字段。
40 个旧配置及历史结果身份保持不变；旧算法主循环和正文均不修改。

## 运行命令

在项目根目录、`smc` 环境中：

```bash
# 预览，不训练、不生成、不创建结果目录
bash gbhaa_experiments/launchers/run_post_objective.sh

# 推荐快速比较：Citeseer，seed 15、16，各 60 个动作，共 24 项
bash gbhaa_experiments/launchers/run_post_objective.sh --execute

# 跑完后审计并汇总
bash gbhaa_experiments/launchers/run_post_objective.sh summary
```

结果：`gbhaa_experiments/post_objective_results`。
报告：`gbhaa_experiments/followup_reports/post_objective`。
底层沿用已有保存的源轨迹（source budget=0.1），实际仅回放其前 60 步，
不是跑完整 10% 攻击。继承当前 Python/Ollama/API 设置。
每项最多 180 次逻辑生成请求；24 项共 4,320 次，缓存命中会减少真实请求。

只想先做最小筛选，可用独立目录运行 12 项：

```bash
bash gbhaa_experiments/launchers/run_post_objective.sh --execute --seeds 15 \
  --output-root gbhaa_experiments/post_objective_seed15
bash gbhaa_experiments/launchers/run_post_objective.sh summary \
  --root gbhaa_experiments/post_objective_seed15 \
  --output-dir gbhaa_experiments/followup_reports/post_objective_seed15
```

中断后，确认同一试验没有仍在运行，再在原命令加 `--retry-failed`。
改代码、环境、种子或计划后应换新 `--output-root`；汇总的 `--root` 要对应。

后续验证可只带基线和胜出配置，保持三个组：

```bash
# 把 objective_margin 替换成这轮实际胜出配置；本命令不预设谁更好
bash gbhaa_experiments/launchers/run_post_objective.sh --execute \
  --profiles transfer_relu objective_margin --seeds 15 16 17 18 19 \
  --output-root gbhaa_experiments/post_objective_confirm
bash gbhaa_experiments/launchers/run_post_objective.sh summary \
  --root gbhaa_experiments/post_objective_confirm \
  --output-dir gbhaa_experiments/followup_reports/post_objective_confirm
```

## 如何判断是否值得保留

- ACC 越低攻击越强。先查看同配置内 `Pre ACC−Post ACC`、
  `Parallel ACC−Post ACC`，再比较新 Post 和 ReLU 基线 Post。
- `paired_comparisons.csv`：逐种子差值、均值、样本 SD 和 95% t 区间。
- `joint_advantage.csv`：每个种子取上述两个差值中的较小值，统计
  **同时严格胜过两个对照**的种子数和最差种子；平局不算胜。
- `objective_effects.csv`：完整四配置时，报告边界目标/局部权重的
  2×2 主效应与交互。显式只跑部分配置时对应完整因子实验不可估计。
- `per_trial.csv`：实际幅度、覆盖/重复、候选档位、局部权重有效目标数
  （`1/sum(w²)`）和回退次数。不同目标的 loss/gain 数值不能直接比较。
- 缺失或审计失败时不推荐赢家，联合优势和因子效应不作完整结果报告。
  排名仍展示可读的已有观测，必须结合完整性状态。

默认两个种子用于快速复查，不能据此宣称稳定优于其他方法；种子 15–19
也已经参与过开发。配对区间仅为未经多重搜索校正的描述性区间。
此次不改变评估标准，也不保证 Post 获胜；独立 ReLU 代理依然没有模拟
victim 的重训练过程，迁移差距仍可能限制改进。
