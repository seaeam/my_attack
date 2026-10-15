# 选择器重跑后复核（2026-10-08）

## 文件状态

- 当前40个 result.json 均存在，其中39个标为complete，1个标为failed。
- 严格汇总：6个完整四臂区组，0个缺失文件，13个拒绝记录；ready_for_full_seed_reporting=false。
- 5% 的有效完整区组仍为 seeds15–16；10% 仍为seeds15–18。

## 新增记录不一致

| 目录seed（5%） | result.seed | 保存的命令seed | attack.seed | 工件与哪个旧seed完全相同 |
|---|---|---|---|---|
|17|17|16|16|16|
|18|18|15|15|15|
|19|19|16|16|16|

以上情况适用于每个seed的全部四种选择器。逐文件SHA256比较确认 attack.json、evaluation.json、rounds.json、combined_predictions.npz 和 stdout.log 完全一致，详见artifact_identity_audit.json。

这12个记录均首先在配置指纹检查处被拒绝（fingerprint mismatch）；内部seed及重复工件也不能支持将它们作为新的独立种子。这里描述的是当前文件内容，不推断这些文件由谁或如何生成。

10% / seed19 / Random 仍为2026-09-27的failed记录，缺少attack.json、evaluation.json和最终预测。

## 对论文的影响

- 与上次核验相比，可用的准确率、配对差值和置信区间完全相同；论文仍使用n=2/4，不扩大样本量。
- 本次没有改写论文或实验结果，没有运行攻击。
- 需核对本次实际运行命令及输出根目录，以定位新的有效结果。
