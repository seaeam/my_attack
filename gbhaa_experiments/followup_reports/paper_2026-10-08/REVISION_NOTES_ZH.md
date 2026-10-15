# 补实验核验与论文修订记录（2026-10-08）

目标：`/Users/shemingcong/Downloads/Thesis/GBHAA/Main-AAAI.tex`。未启动实验，未修改攻击代码或旧结果。

## 核验结论

- Factorial：40/40 运行有效，10 个完整四臂区组；两预算均为 seeds 15–19。预测重算、身份、协议、源文件与输入来源、结构动作回放、规定配对均通过。
- Selectors：计划 40 个运行中，27 个运行有效，但仅 24 个属于完整四臂区组；5% 使用 seeds 15–16（n=2），10% 使用 seeds 15–18（n=4）。9 个结果文件缺失，4 个已有记录被拒绝。论文明确标为 preliminary。

## 可支持的结论

- 双阶段属性加权的攻击后准确率均值最低：5% 为 58.79±0.61%，10% 为 57.01±1.79%；双 topology 为 59.58±1.36% 和 57.89±1.19%。
- 配对局部加权收益在全局加权下为 1.31/0.65 pp；交互为 1.33/0.56 pp，两个交互 CI 都跨零。十项计划对比的 Holm p 均不低于 0.625。不能写显著优势或交互已被证实。
- 已有 selector 完整区组中，5% Random 均值最低，10% 普通 PR 均值最低；没有 AE-PR 优越性的证据。不同预算 seed 集合不同，不据此推断跨预算趋势。

## 实际协议与解释限制

- 两套实验均在翻边前选择属性，并冻结结构特征；不能验证修改后图的必要性，也不能代表局部属性变化反馈到结构搜索的完整耦合。
- 全局为 degree source、P @ z、含初态 30 项累积；局部为双端点源、P.T @ s、60 次更新后读出。2×2 仅切换属性边权；普通 PR 比较还改变源分布和读出。
- 40 个 factorial 的 10,980 轮记录均非全同分，独立分数数量 1,840–2,011；同分仍存在。此为保存日志诊断，不是独立回放评分。
- Citeseer 实际使用 DeepRobust nettack 划分。meta.py 虽接收 normal 参数，在该数据分支硬编码 nettack；已比对源文件哈希。图 2,110 节点，评价 1,899 节点。
- 所有纳入区组均最终改动 2,110 节点，21,100 次访问。各臂实际净修改条目不相等，见 modification_counts.csv；计数来自运行记录。
- 使用 feature_i fallback 词表，仅能作为特征空间诊断。未保存最终特征矩阵或范数幅度，不能恢复最终修改幅度或证明自然文本语义保持。
- Selector 保留原全局 NumPy 排序，factorial 使用 node-ID 精确并列规则，两个协议独立汇总。

## 拒绝记录

- 0.05 / seed 17 / ae_pr: Run incomplete/failed
- 0.05 / seed 18 / ae_pr: Stored seed differs from directory/job
- 0.05 / seed 19 / ae_pr: Stored seed differs from directory/job
- 0.1 / seed 19 / random: Run incomplete/failed

## 缺失运行

- 0.05 / seed 17 / pr
- 0.05 / seed 17 / degree
- 0.05 / seed 17 / random
- 0.05 / seed 18 / pr
- 0.05 / seed 18 / degree
- 0.05 / seed 18 / random
- 0.05 / seed 19 / pr
- 0.05 / seed 19 / degree
- 0.05 / seed 19 / random

## 论文修改

- 替换原单 seed 2×2 图为五种子图；新增配对条件效应/交互表。
- 替换原未包含 AE-PR 的不匹配选择器表为真实四臂初步结果，明确 n 和未完成状态。
- 方法补充 degree 配置和非退化日志证据；contribution、Conclusion 同步限定证据范围。
- 新增协议与实际消耗附录；未更改主算法的翻边顺序、主表、原有串行/并行或效率结果。
- 原 tex 与 PDF 已备份至论文目录 revision_backups/2026-10-08-followup/。

## 数据与复现

- factorial/summary.json：完整矩阵与十项计划检验；per_seed.csv 保留逐 seed 精确指标。
- selectors/summary.json：计划双预算的汇总及完整缺失/拒绝清单（ready_for_full_seed_reporting=false）。
- selectors_005 与 selectors_010：按单预算的独立核验报告；论文不将它们当作完整五种子矩阵。
- paper_data_audit.json、modification_counts.csv：排名诊断、实际消耗与有效划分核对。
- [绘图脚本](/Users/shemingcong/Downloads/my_attack/docs/experiments/build_paper_assets_20261008.py)：用 smc Python 重建新图和核验附属数据；不运行攻击。该脚本放在实验源码扫描范围之外，避免论文绘图辅助代码改变实验源码指纹。

## 独立核对

- 定向审查已逐项复算新图、准确率、配对 CI、clean 均值与实际条目表；未发现数值或方向错误。
- 本机 latexmk/pdflatex + BibTeX 编译成功，PDF 共 15 页；新增内容位于第 11–12、14 页。已渲染并目视检查第 11–14 页，新图和三张表无溢出、重叠或截断，最终日志无 Overfull。
- 原主结果表 table/Exp-1_mainResults.tex:66 的 \cite{xxx} 仍产生未定义引用警告；未为它捏造或替换引用。
- 内置编译器下载 LaTeX 依赖时超时，本次 PDF 使用本机工具链生成。
- 已保存 Main-AAAI.patch，并更新论文目录 Main-AAAI.pdf；原始 tex/PDF 备份仍保留。
