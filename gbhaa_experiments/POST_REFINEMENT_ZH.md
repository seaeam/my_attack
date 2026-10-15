# Endpoint-Post 第二轮：围绕 alpha=0.50 的匹配调参

## 目的与当前证据

上一轮 alpha050 与 baseline 的验证 ACC 同为 66.3507%，仅 NLL 更高。当前不能认定 alpha050 已获得稳定 ACC 改善。

本轮保留这两个参考配置，新增 8 个邻近或组合配置，共 10 个。所有配置都运行 Post、Pre、Parallel，相同参数下比较三组。主配置、旧结果和论文不变。

针对保存的 alpha050 Post 属性状态做过一次离线诊断：同一状态下，仅把扩散图从 pre 换成 post，alpha050 的目标节点集合在 3/60 轮变化；`a50_diffused` 为 30/60，`a50_f03_diffused` 为 41/60。后两项因此值得优先关注。

该诊断复用了旧属性和访问历史，没有重新生成各候选的攻击轨迹，也没有运行 victim 或使用标签。**对图变化更敏感不等于攻击 ACC 更低**，最终以实际验证结果为准。诊断原始记录在 `followup_reports/post_tuning/refinement_sensitivity.json`。

## 1．十个候选

| 配置 | alpha | 边权下限 | 访问惩罚 gamma | 读出 |
| --- | ---: | ---: | ---: | --- |
| baseline | 0.28 | 0 | 0 | last |
| alpha050 | 0.50 | 0 | 0 | last |
| alpha040 | 0.40 | 0 | 0 | last |
| alpha060 | 0.60 | 0 | 0 | last |
| a50_f01 | 0.50 | 0.01 | 0 | last |
| a50_f03 | 0.50 | 0.03 | 0 | last |
| a50_f05 | 0.50 | 0.05 | 0 | last |
| a50_diffused | 0.50 | 0 | 0 | diffused |
| a50_f03_diffused | 0.50 | 0.03 | 0 | diffused |
| a50_coverage025 | 0.50 | 0 | 0.25 | last |

共同设置：60 次扩散迭代；全局 degree source；每轮 4 次成功节点属性写入，每节点最多访问 10 次；生成上限 50 tokens；相同投影、结构动作和 victim 协议。

- 下限用于所有已有边：`w = floor + (1-floor)*cosine`，防止余弦为零的边完全失去传播权重。
- `diffused = max(last_state - alpha*q, 0)`：扣除直接重启项，让排名更多反映传播到的质量。
- 惩罚为 `score / (1+visits)^gamma`，仅对一个候选使用温和的 gamma=0.25。
- 下限和读出属于已明确记录的算子变体；不会静默覆盖原 AE-PR 定义。选择器仍只接收本组的一张图，不把 pre/post 差值作为排名输入。
- 每组后续分组/生成统一使用 post 图，各自累积属性修改。Parallel 仍是 pre 图上的独立重启点扩散。

原 `quick`、`tiny`、`extended` 的配置集合和参数保留；新预设名为 `refine`。直接调用通用 runner 时须加 `--matched-controls`；下方专用脚本已经设置。

## 2．第一阶段：匹配筛选

工作量：Citeseer，seed 15，每次 60 个结构动作；10 配置 × 3 组 = **30 次短试验、1,800 次动作回放**。不运行完整 10% 攻击。时间受缓存与生成服务影响。

使用原先准备好的 `smc` 环境和本地生成服务：

```bash
cd /Users/shemingcong/Downloads/my_attack
export PYTHON_BIN=/Users/shemingcong/anaconda3/envs/smc/bin/python

# 预览，默认不训练、不调用生成
bash gbhaa_experiments/launchers/run_post_refinement.sh

# 正式执行这轮短筛选
bash gbhaa_experiments/launchers/run_post_refinement.sh --execute

# 完成后汇总
bash gbhaa_experiments/launchers/run_post_refinement.sh summary-screen
```

脚本沿用现有环境规则：显式 PYTHON_BIN 优先，其次当前 Conda 环境；默认本地 Ollama 地址、模型及占位 key 与旧启动器相同，已有环境变量可覆盖。

新结果目录：`gbhaa_experiments/post_refinement_results`。
新报告目录：`gbhaa_experiments/followup_reports/post_refinement`。
**请使用新脚本和新目录**；代码更新后继续执行旧结果目录的旧计划会触发源码指纹不匹配。旧结果仍可汇总，不需要删除。

筛选按 Post 验证 ACC 最低、ACC 并列时 NLL 最高选配置，保留所有候选和三组结果。不会根据“哪个对照更弱”挑配置，也不会因为 Post 未领先而删除某个种子。

## 3．第二阶段：固定候选后复核

必须先完成第一阶段汇总，生成有效 `recommendation.json`。第二阶段自动读取该推荐并冻结参数：

```bash
# 先预览
bash gbhaa_experiments/launchers/run_post_refinement.sh confirm

# 固定候选，运行其他四个开发种子
bash gbhaa_experiments/launchers/run_post_refinement.sh confirm --execute

# 汇总配对差值及区间
bash gbhaa_experiments/launchers/run_post_refinement.sh summary-confirm
```

默认 seeds 16–19，每次 120 个动作；胜出配置运行三组，并保留 baseline Post：通常 **16 次试验、1,920 次动作回放**。若 baseline 胜出，则是 **12 次试验、1,440 次动作回放**。不重跑全部候选。

这些 seeds 的历史结果已被观察过，因此属于开发复核，不是全新的独立测试。60 步与 120 步分别汇总，不能拼成一个五种子均值。完整预算或跨数据集稳定性仍需后续固定配置验证；本轮不自动启动。

复核结果：`gbhaa_experiments/post_refinement_confirm_results`。
复核报告：`gbhaa_experiments/followup_reports/post_refinement_confirm`。

## 4．怎样判断优势

主要看 `summary.md` 的 **Paired ACC comparisons** 及 `paired_comparisons.csv`：

- 差值定义为 `对照 ACC − Post ACC`，正数表示 Post 攻击更强。
- 分别看对 Endpoint-Pre 和 Parallel 的同配置比较。
- `accuracy_wins/losses/ties` 只按 ACC 判断。ACC 相同、只有 NLL 变好，仍是 ACC 平局。
- 95% 区间基于配对种子差值，使用样本标准差和 `df=n−1` 的 t 分布。单种子不提供区间。
- 区间为未做多重搜索校正的描述性开发区间，不能因筛选阶段某个区间为正便宣布普遍稳定领先。
- 核对有效/观测/计划种子数以及比较、整体矩阵的完成状态。缺失或拒收保留记录并抑制推荐，不能只选成功运行。拒收区组的区间不展示，有效种子数为零；保留的差值只用于审计。部分完成的区组也检查已有结果是否匹配。
- 同时观察 Post 自身是否比 baseline 更强，及实际修改节点数、条目数和幅度。相同写入预算不保证实际幅度完全相同。

若固定候选在后续种子对两种对照都大多获胜、配对区间位于零以上，才有更充分的稳定优势证据；如果仍有明显反转，应保留该结果并调整结论。

## 5．续跑和自定义目录

同一源码、环境和计划下，重复命令会跳过已通过验证的完整结果。失败重试保留原尝试：

```bash
bash gbhaa_experiments/launchers/run_post_refinement.sh --execute --retry-failed
bash gbhaa_experiments/launchers/run_post_refinement.sh confirm --execute --retry-failed
```

运行命令允许转发原 runner 参数。若修改输出目录，汇总须对应传 `--root`、`--output-dir`，复核须传对应 `--recommendation`。不同种子、步数或配置矩阵使用新的结果目录。

本轮只新增代码、离线诊断和运行说明，没有执行新攻击或修改论文。
