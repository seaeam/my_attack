# Endpoint-Post / Endpoint-Pre / Parallel 实现审查

审查日期：2026-10-13。本轮未修改实验代码或论文，未运行攻击、训练或文本生成。

## 结论

当前补实验和快速调参入口确实执行三种不同的局部选择规则。代码审查、离线测试和实际结果的逐轮复算一致；未发现 Pre/Post 图传反、翻边破坏 pre 快照，或 Parallel 直接使用当前结构端点的问题。

| 组别 | 扩散图 | 重启节点 | 选点后的属性分组与生成 |
| --- | --- | --- | --- |
| Endpoint-Post | 本轮翻边后的图 | 当前结构动作的两个端点 | 使用翻边后的图 |
| Endpoint-Pre | 本轮翻边前的图 | 同一结构动作的两个端点 | 使用翻边后的图 |
| Parallel | 本轮翻边前的图 | 由 seed 和轮次独立抽取的两个不同节点 | 使用翻边后的图 |

三组复用同一保存的结构动作序列，初始图、初始属性、训练划分和生成规则一致；基线每轮选取 4 个合法节点，最多访问每个节点 10 次。候选排序覆盖全图，然后使用同一访问次数约束过滤。

## 调用链与关键证据

- [调参运行命令](/Users/shemingcong/Downloads/my_attack/gbhaa_experiments/run_post_tuning.py:38) 调用专用 `meta_post_tuning` 入口。
- [入口注册](/Users/shemingcong/Downloads/my_attack/gbhaa_experiments/entrypoints/meta_post_tuning.py:209) 将模型类替换为 `TuningAttack`；随后复用 `meta.py` 的初始化，再调用继承的动作回放流程。
- [回放循环](/Users/shemingcong/Downloads/my_attack/gbhaa_experiments/entrypoints/meta_action_state.py:184) 保存 pre，并通过复制创建 post；随后选择目标、进入 post 状态并生成属性。
- [三组选择分支](/Users/shemingcong/Downloads/my_attack/gbhaa_experiments/entrypoints/meta_post_tuning.py:117) 明确把 pre 或 post 传给评分函数；Parallel 单独产生重启点。原始补实验入口采用相同分支规则。
- [独立重启点](/Users/shemingcong/Downloads/my_attack/gbhaa_experiments/action_state_protocol.py:49) 仅依赖节点数、seed 和轮次，允许与端点偶然重合。
- [局部扩散](/Users/shemingcong/Downloads/my_attack/gbhaa_experiments/post_tuning_protocol.py:213) 使用传入图上的属性相似度权重，以及 `alpha*q + (1-alpha)*(P.T@state)` 更新。这里的局部 q 均匀分配到两个重启节点；全局结构模块的 degree source 是另一项配置。

代码虽然为每一组都计算 pre/post 诊断分数，但最终选点只使用相应组的排序。候选过滤不改变访问历史，诊断分支不会把 Post 的候选传给 Pre 或 Parallel。

## 实际运行验证

审查结果根目录：`gbhaa_experiments/post_tuning_results`，Citeseer / seed 15 / 60 个结构动作。

- 保存计划中的 66 个源码文件哈希全部与当前源码一致。
- 重新运行 32 项相关离线测试，全部通过，覆盖翻边快照、增删边、三组分支、基线评分一致性、预算及验证集评估。
- 使用保存的初始属性、逐轮属性写入及结构动作，重建每轮实际输入并重新计算目标节点：

| 运行 | 验证轮数 | 验证节点写入次数 | 复算选点不一致 |
| --- | ---: | ---: | ---: |
| baseline / Endpoint-Post | 60 | 240 | 0 |
| baseline / Endpoint-Pre | 60 | 240 | 0 |
| baseline / Parallel | 60 | 240 | 0 |
| alpha050 / Endpoint-Post | 60 | 240 | 0 |
| 合计 | 240 | 960 | 0 |

同时核对了每轮 pre/post 图、实际扩散图、重启节点、当前属性哈希、分组图及最终图和属性。三组基线和 alpha050 的结构轨迹完全相同。Parallel 的 60 对独立重启点与当轮端点共偶然重合 1 个节点，符合其抽样定义。

## 解释时必须保留的边界

1. **Pre/Post 专指扩散选点所用的图。** 三组的后续分组嵌入和属性生成都使用 post 图。这是一项局部选择器机制对照，不是把属性生成的全部环节分别放在翻边前后。
2. **Parallel 的具体定义是独立重启点扩散。** 最终目标节点仍由 AE-PR 排序产生，不是直接随机抽取四个目标；它仍使用共享结构历史形成的 pre 图，因此不能描述为整个属性攻击流程完全独立于结构攻击。
3. **三组只保证初始属性相同。** 后续各自累积属性修改，因此每轮的当前属性可能不同。这属于固定结构轨迹下的累积效果比较；逐轮纯粹的“只换图、保持同一属性状态”应看同一组内的 pre/post probe。实际基线 Post 与 Pre 从第 6 轮开始出现不同的输入属性状态。
4. **快速调参只有基线三组采用同一配置。** 当前 alpha050 仅用于 Post，Pre 和 Parallel 仍为 alpha=0.28；这项跨配置比较不能单独归因于 post 图。代码已有 `--matched-controls`，可在相同配置下运行三组。
5. **旧主循环仍保持原顺序。** 按此前要求，`heir.py` 的旧主循环保留翻边前的属性选择。此次补实验通过专用回放实现 Post，不能据此认定仓库所有旧入口都已变为 Post。结构轨迹生成调用旧循环时关闭属性攻击。

## 为什么选点差异可能不大

基线 Post 的同状态诊断（同一轮当前属性、端点、访问历史，仅改变 pre/post 扩散图）：

- 44/60 轮扩散分数发生变化；
- 24/60 轮前四名的有序列表发生变化；
- 5/60 轮前四名的节点集合发生变化；
- 16/60 轮动作两端的属性相似度为零，且扩散分数差为零。基线 `weight_floor=0` 时，这条边在加权图中没有传播权重，翻动它不会改变扩散。

例如第 5 轮添加边 `[1774,185]`，同属性状态下 Pre 选择 `[1774,185,1781,1791]`，Post 选择 `[1774,185,1791,1614]`。这证明图变化确实能影响选择，但不是每轮都能改变最终节点集合；排序变化也可能影响后续生成分组，不能把“集合相同”直接视为“全部处理相同”。

这些证据支持当前三组分支按定义执行，不构成 Endpoint-Post 必然优于其他组的保证，也不等于对所有旧实验结果的重新验证。
