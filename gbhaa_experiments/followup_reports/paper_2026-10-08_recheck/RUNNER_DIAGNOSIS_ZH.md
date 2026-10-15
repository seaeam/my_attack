# 运行器身份检查诊断（2026-10-08）

## 已查明

- make_jobs → build_command → meta_followup.main → meta.py 参数转发链路在离线检查中正确保留 seed 17、18、19 及各自输出目录；没有训练模型或调用 LLM。细节见 seed_dispatch_trace.json。
- 当前 selector 完成检查存在漏洞：运行记录写 seed17 而独立 attack.json 写 seed16 时，旧 load_completed_artifacts 仍返回成功。最小临时工件复现与真实现有目录都确认此问题。
- 当前 seed17、18、19 的保存命令除了 seed 为16、15、16，--followup-artifacts 路径也指向相应旧seed目录。正常命令构建测试没有复现这一错误。因此尚未查明这些旧命令和工件进入新目录的来源，不能把身份校验漏洞直接说成重复文件产生的根因。

## 已修改

- gbhaa_experiments/run_followup.py：所有协议在完成/恢复时核对嵌入与独立工件，核对 seed、dataset、selector、local_graph、global_transition、global_source。
- 启动子进程前保存实际命令，终端显示 seed/selector/budget；异常保存 error_type，空消息用异常类型补充。
- tests/test_followup_runner.py：加入错误工件的临时文件复现、三seed派发路径、主循环拒绝错误seed和中断记录测试。
- 上次论文绘图辅助脚本移至 docs/experiments/build_paper_assets_20261008.py，避免生成报告辅助代码进入实验源码指纹。

## 验证

- 修复前新回归测试有4项按预期失败；修复后通过。
- runner、summary、recording、factorial 相关测试合计52项通过。
- 修复后读取已有 selector seed16成功，seed17–19均明确拒绝result与独立工件不一致。
- 单任务新目录dry-run通过；不会开始攻击。计划保存为identity_check_plan.txt，实际执行命令见FOLLOWUP_EXPERIMENTS_ZH.md开头。

## 仍需定位

- 现有重复结果的来源未复现；需要在新的输出目录记录一次真实调用，或取得此前实际执行的完整命令/终端日志。
- 源码指纹已改变，新定位输出不应与旧cohort直接合并。定位命令仅为单个任务，不是完整矩阵。
- 没有改写原始实验结果、扩大论文样本量或重跑长时间攻击。
