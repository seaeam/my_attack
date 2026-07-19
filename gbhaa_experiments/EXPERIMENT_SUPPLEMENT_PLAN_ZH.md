# GBHA 完整补充实验方案与运行手册

本文档基于当前 `gbhaa_experiments` 中的实际代码，目的是把七项补充实验落实为可以执行、检查、汇总并写入论文的完整流程。代码目录说明见 [`README.md`](./README.md)。

本文档回答四个问题：每项实验验证什么、怎样保持公平对照、当前代码如何运行、什么样的结果才可用于论文。

## 一、实验总目标

| 论文主张 | 对应实验 | 必要证据 |
|---|---|---|
| Granular-Ball 粗化降低搜索代价并保留有效方向 | 端到端效率、GB 消融、粗粒度梯度一致性 | 时间/内存、攻击效果、粗细梯度排序一致性 |
| 结构与属性协同优于单一或完全独立攻击 | Hybrid Coupling 消融 | `serial` 与 `edge_only`、`feature_only`、`parallel` 的配对结果 |
| LLM 及其约束组件各自有效 | LLM 消融 | 完整方法、无 LLM 和各组件移除版本的比较 |
| AE-PPR 与约束改善传播及隐蔽性 | AE-PPR 2×2、隐蔽性分析 | 全局/局部贡献和效果—隐蔽性权衡 |

不要预先假定 GB、`serial` 或完整 LLM 版本一定最好，也不要只保留符合预期的随机种子。

## 二、统一实验协议

### 2.1 数据集

效率实验固定使用 Citeseer 和 Cora。其他文本或小图实验组仍使用 Citeseer；GB 粗化消融保留原协议中用于结构对照的 Pubmed 任务。

| 实验 | 数据集 |
|---|---|
| 效率 | Citeseer、Cora |
| GB 消融 | Citeseer、Pubmed |
| Hybrid、LLM、梯度一致性、AE-PPR、隐蔽性 | Citeseer |

效率矩阵不包含 Pubmed 或 CS，不得将该协议报告成大图扩展性实验。

### 2.2 种子与预算

正式矩阵统一使用种子 `15, 16, 17, 18, 19`。

- 端到端效率：1%、5%、10%；
- 其余带预算实验：5%、10%；
- 梯度一致性：比较 8、16、32 个粒球，不执行完整攻击预算。

同一张表中的方法必须使用相同数据划分、种子和预算。

### 2.3 固定参数

效率实验不套用一组通用调参。两个数据集只共享：攻击模型 `Meta-Both`、数据划分 `normal`、训练 200 轮、每步扰动数 1，以及 `--freeze_structure_features` 固定结构搜索所用的原始特征。其余设置只复制根目录 [`run_citeseer_gb.sh`](../run_citeseer_gb.sh) 和 [`run_cora.sh`](../run_cora.sh) 中列出的下列数据集专用参数值：

- Citeseer：层次深度 4，内部迭代 60，学习率 0.05；全局重要节点比例 0.45，全局 PPR 为 0.08/120，种子策略 `degree`；单节点最多访问 4 次，LLM 重试 3 次，每节点保留特征词 35，局部候选比例 0.08，局部 PPR 为 0.26/30，属性簇大小 2–5，最低余弦相似度 0.65，类别判别特征数 14，簇属性特征数 14，最大新增特征数 40。
- Cora：层次深度 2，内部迭代 450，学习率 0.012；全局重要节点比例 0.25，全局 PPR 为 0.08/180，种子策略 `uniform`；单节点最多访问 16 次，LLM 重试 3 次，每节点保留特征词 80，局部候选比例 0.12，局部 PPR 为 0.28/60，属性簇大小 2–4，最低余弦相似度 0.35，类别判别特征数 36，簇属性特征数 36，最大新增特征数 70。

两个数据集的效率入口始终统一为 `meta.py`。`run_cora.sh` 当前的实际执行行可能是 `meta_gsage.py`，但效率协议不复制该入口，只复制上述明列参数值。

其他实验保留原协议的通用配置：`Meta-Both`、`normal`、200 轮、层次深度 2、每步扰动 1、内部迭代 10、学习率 0.01、全局重要节点比例 0.10、全局 PPR 0.15/30、种子策略 `degree`，并固定结构搜索特征。其属性设置为：每节点保留特征词 15、局部候选比例 0.05、局部 PPR 0.20/25、属性簇大小 2–8、最低余弦相似度 0.85、类别判别特征数 10、簇属性特征数 10、最大新增特征数 20、单节点最多访问 1 次、LLM 重试 0 次。消融实验只改变其名称对应因素，其他条件保持一致。

### 2.4 特征词表边界

默认为严格对齐模式：缓存词表长度与特征维数不一致时直接停止，除非启动器显式选择部分非空缓存词表或回退占位词表。

效率实验对 Citeseer 和 Cora 显式传入 `--allow_partial_vocabulary`，使用各自 `bow_cache/*.pkl` 中的 500 个可读词项，不生成 `feature_i`，也不传入 `--allow_fallback_vocabulary`。当前代码仅保留缓存顺序并按位置映射到最前面 500 个特征列（Citeseer 共 3,703 维，Cora 共 1,433 维），没有验证缓存词项与数据集原始特征名在语义上同一。当前属性写回行为按以下顺序执行：

1. LLM 文本在 500 词缓存上向量化，然后按原有前导列对应写入完整特征宽度，其余列补 0；
2. 按 `--text_max_added_words` 删减超过上限的新增词项；
3. 若新向量与原始完整特征的余弦相似度低于 `--text_similarity_min`，就二分搜索两者的线性插值比例，将向量投影回相似度下限。

因此，补零后的生成向量在后续特征列上为 0，但第 3 步与原始特征插值时可能按比例恢复后续列。该路径可以表述为遵循历史脚本的可读缓存词表路径，但不得报告为缓存词项与前 500 个原始特征名已语义对齐、全 3,703/1,433 列具有真实词语义，或自然语言语义攻击有效。

其他仍显式传入 `--allow_fallback_vocabulary` 的启动器属于离散特征空间消融。其 `feature_i` 只与特征列对齐，不可作为真实单词、自然语言语义保持或文本流畅性的证据。

### 2.5 效率计时与资源测量协议

正式效率矩阵开始前，先用与矩阵完全相同的模型做一次预热：

```bash
ollama run "$OLLAMA_MODEL" "只回复 OK"
```

预热完成后立即开始 90 个运行，按启动器命令顺序串行执行，不并发其他实验。全矩阵期间保持同一 Ollama 服务进程、同一 `OLLAMA_MODEL` 和同一服务状态，不更换代码或 provider 默认值。当前 API 生成上限为 `max_tokens=50`，不为 `gb`、`kmeans` 或 `node_level` 单独改解码设置。

墙钟时间覆盖整条攻击命令，包括阻塞等待 LLM 的延迟。只有 `/usr/bin/time` 存在且其输出可解析时，`peak_rss_mib` 才会有值；否则为空。该峰值 RSS 只覆盖攻击命令进程，不包括单独运行的 Ollama 进程或其 GPU 显存，论文表格必须明示该口径。

`result.json` 的 `runtime_environment` 会记录 `OLLAMA_MODEL`，保存的脱敏命令会记录 API base URL。为便于完整审计，研究者还应在结果目录旁保存 `ollama --version` 的输出。

当前输出中的 `LLM Calls` 是 cluster-level 模板生成缓存未命中/生成调用计数，`Cache Hits` 是对应的模板缓存命中数。传输/API 重试不会单独增加 `LLM Calls`，所以该字段不是 HTTP 请求数或成本计数；需要实际请求数时，必须使用 Ollama 或其他 provider telemetry。

### 2.6 统计规则

正式矩阵计划每组运行种子 `15, 16, 17, 18, 19`。汇总代码会先排除失败记录，并对每个数据集/变体/预算/种子只保留最新的成功记录，因此某组有效样本数可能 `n<5`。正式表格只有在逐组确认 `n=5` 后才能将该行视为完整矩阵结果，且每组都必须披露 `n`。表格同时给出数据集、预算、变体、Clean Accuracy、Attacked Accuracy、Accuracy Drop 和 `mean ± std`；不得笼统声称聚合会自动对五个种子统计。

当前代码不自动计算显著性检验。没有另行完成配对统计检验时，不要声称“具有统计显著性”。

## 三、完整矩阵规模

| 实验 | 攻击子任务 | 需要外部 LLM 的子任务 |
|---|---:|---:|
| 效率 | 90 | 90 |
| GB 消融 | 70 | 0 |
| Hybrid 消融 | 40 | 30 |
| LLM 消融 | 60 | 50 |
| AE-PPR 对比 | 40 | 40 |
| 隐蔽性 | 40 | 30 |
| 粗粒度梯度一致性 | 1 个诊断进程，内部含 3 种粒球数 | 0 |

完整方案为 340 个攻击子任务和 1 个梯度诊断，其中 240 个攻击子任务会访问 OpenAI 兼容接口。“LLM 子任务”表示该命令启用外部 LLM，不表示只发出一次 HTTP 请求；一个子任务可能有多个 cluster-level 模板生成缓存未命中，而每次生成内部的 API 重试又可使实际 HTTP 请求数高于 `LLM Calls`。

不要并发执行七个正式启动器。效率实验的 90 个子任务必须串行，否则时间和内存不可比。

## 四、环境准备

所有命令从仓库根目录执行。启动脚本默认使用当前激活环境中的 Python；也可以先显式记录该解释器，供后续直接调用 Python 模块：

```bash
export PYTHON_BIN="$(command -v python)"
```

### 4.1 安装依赖

```bash
"$PYTHON_BIN" -m pip install -r requirements.txt
```

安装后执行：

```bash
"$PYTHON_BIN" - <<'PY'
import torch
import torch_geometric
import torch_scatter
import torch_sparse
import deeprobust
import openai
import httpx
import transformers
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("all required imports: OK")
PY
```

只有完整输出 `all required imports: OK` 才开始正式实验。

当前机器在 2026-07-17 的实际预检状态是：Python 3.13.13、PyTorch 2.11.0 CPU 已安装；`torch_geometric`、`torch_scatter`、`torch_sparse`、`openai` 和 `transformers` 尚未安装；`pip install --dry-run -r requirements.txt` 已确认存在匹配的可安装版本。因此当前环境必须先安装依赖。

### 4.2 检查数据

```bash
test -f Data/citeseer.npz && echo "Citeseer: OK"
test -f Data/cora.npz && echo "Cora: OK"
test -f bow_cache/citeseer.pkl && echo "Citeseer vocabulary: OK"
test -f bow_cache/cora.pkl && echo "Cora vocabulary: OK"
du -sh Data bow_cache
```

效率矩阵必须同时具备 `Data/citeseer.npz`、`Data/cora.npz`、`bow_cache/citeseer.pkl` 和 `bow_cache/cora.pkl`。正式运行前只读打印两个缓存的词表长度和头部词项，并记录文件校验和：

```bash
"$PYTHON_BIN" - <<'PY'
import pickle
from pathlib import Path

for path in (Path("bow_cache/citeseer.pkl"), Path("bow_cache/cora.pkl")):
    with path.open("rb") as handle:
        vectorizer = pickle.load(handle)
    vocabulary = list(vectorizer.get_feature_names_out())
    print(f"{path}: length={len(vocabulary)}, head={vocabulary[:10]}")
PY
shasum -a 256 bow_cache/citeseer.pkl bow_cache/cora.pkl
```

只有两个长度都确认为 500 时，才与本协议相同。应将输出的校验和、头部词项和缓存文件来源说明一起保存到结果旁；这些检查只能确认长度和文件身份，不能验证缓存词项与前 500 个原始特征名语义同一。GB 消融中的 Pubmed 任务仍可通过 DeepRobust 加载路径获取。

### 4.3 准备 Ollama

LLM 启动器默认使用 `http://localhost:11434/v1` 和 `llama3.2:1b-instruct-fp16`。

在独立终端启动：

```bash
ollama serve
```

模型不存在时执行：

```bash
ollama pull llama3.2:1b-instruct-fp16
```

实验终端执行：

```bash
curl -fsS http://localhost:11434/api/tags
export OLLAMA_MODEL=llama3.2:1b-instruct-fp16
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
```

`OPENAI_API_KEY=ollama` 只是本地兼容接口的非敏感占位值。当前机器已安装 Ollama 且模型存在，运行时仍需确认服务没有退出。

### 4.4 环境一致性

正式实验期间不要在种子之间更换 Python 环境、PyTorch/PyG 版本、CPU/GPU、Ollama 模型、量化版本或并发数量。运行器会把可获取的环境信息写入 `result.json`。

## 五、运行前验证

```bash
"$PYTHON_BIN" -m py_compile \
  gbhaa_experiments/run_matrix.py \
  gbhaa_experiments/runner.py \
  gbhaa_experiments/specs.py
"$PYTHON_BIN" -m compileall -q gbhaa_experiments text_attack_generator.py
bash -n gbhaa_experiments/launchers/*.sh
```

预期：指定 Python 文件和实验目录编译通过，Shell 语法检查无错误。

在正式运行前先预览效率矩阵，显式设置 `EXECUTE=0`：

```bash
EXECUTE=0 bash gbhaa_experiments/launchers/run_efficiency.sh
```

预览应精确打印 90 条 `meta.py` 命令：Citeseer 和 Cora 各 45 条。所有命令都应包含 `--use_text_attack`、`--allow_partial_vocabulary` 和外部 LLM 配置，不得包含 `--allow_fallback_vocabulary`、`meta_edge_only.py`/edge-only 入口、Pubmed 或 CS。

## 六、低成本烟测

烟测只检查流程，不作为论文结果。

### 6.1 无 LLM 结构烟测

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_matrix \
  --experiment gb_ablation --datasets citeseer \
  --seeds 15 --ptb-rates 0.001 --variants gb \
  --python "$PYTHON_BIN" \
  --output-root gbhaa_experiments/smoke_results \
  --execute -- \
  --model Meta-Both --split_data normal --epochs 5 \
  --coarsen_method gb --level 2 --step 1 --miter 1 --lr 0.01 \
  --global_important_ratio 0.10 --global_ppr_alpha 0.15 \
  --global_ppr_iters 5 --global_seed_strategy degree
```

通过条件：退出码为 0，生成 `result.json` 与 `stdout.log`，包含 Clean Accuracy 和 Edge attack accuracy，且命令与日志都不含文本攻击或外部 LLM 调用。该烟测只验证 GB 结构消融入口，不验证效率矩阵的联合攻击行为。

### 6.2 确定性无 LLM 属性烟测

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_matrix \
  --experiment llm_ablation --datasets citeseer \
  --seeds 15 --ptb-rates 0.001 --variants deterministic_no_llm \
  --python "$PYTHON_BIN" \
  --output-root gbhaa_experiments/smoke_results \
  --execute -- \
  --model Meta-Both --split_data normal --epochs 5 \
  --coarsen_method gb --level 2 --step 1 --miter 1 --lr 0.01 \
  --global_important_ratio 0.10 --global_ppr_alpha 0.15 \
  --global_ppr_iters 5 --global_seed_strategy degree \
  --freeze_structure_features --allow_fallback_vocabulary \
  --text_attack_max_visits 1 --text_retries 0
```

通过条件：日志显示 3,703 个对齐占位特征、`External LLM Calls: 0`，并产生 Clean、Edge、Feature 和 Combined Accuracy。

### 6.3 本地 LLM 烟测

```bash
OPENAI_API_KEY=ollama \
OLLAMA_MODEL=llama3.2:1b-instruct-fp16 \
"$PYTHON_BIN" -m gbhaa_experiments.run_matrix \
  --experiment llm_ablation --datasets citeseer \
  --seeds 15 --ptb-rates 0.001 --variants full \
  --python "$PYTHON_BIN" --llm-type gpt \
  --api-base-url http://localhost:11434/v1 \
  --output-root gbhaa_experiments/smoke_results \
  --execute -- \
  --model Meta-Both --split_data normal --epochs 5 \
  --coarsen_method gb --level 2 --step 1 --miter 1 --lr 0.01 \
  --global_important_ratio 0.10 --global_ppr_alpha 0.15 \
  --global_ppr_iters 5 --global_seed_strategy degree \
  --freeze_structure_features --allow_fallback_vocabulary \
  --text_attack_max_visits 1 --text_retries 0 \
  --text_budget_per_node 15 --text_topk_ratio 0.05 \
  --text_ppr_alpha 0.20 --text_ppr_iters 5 \
  --text_min_cluster_size 2 --text_max_cluster_size 8 \
  --text_similarity_min 0.85 --text_cdl_topk 10 \
  --text_cluster_attr_topk 10 --text_max_added_words 20
```

通过条件：日志包含 `Detected Ollama`、`Text attack generator initialized` 和 LLM Calls，不包含 `Text attack requested but not available` 或 `Text attack not available, skipping feature attack`。

若外部 LLM 变体的 `metrics.llm_calls` 为 0，先检查候选节点、生成错误和跳过日志，不要直接接受该结果。该字段仍只是 cluster-level 模板生成计数，不是 HTTP 请求数。

### 6.4 梯度一致性烟测

```bash
"$PYTHON_BIN" -m gbhaa_experiments.coarse_gradient_consistency \
  --dataset citeseer --seed 15 --clusters 8 \
  --epochs 5 --miter 1 --global-ppr-iters 5 --topk 3 \
  --max-nodes 200 --no-cuda \
  --output-dir gbhaa_experiments/smoke_results/coarse_gradient
```

应生成 `pairs_k8.csv`、`summary_k8.json` 和 `summary.json`。`--max-nodes` 只用于烟测，正式结果必须使用完整图。

## 七、正式运行顺序

阶段 A：结构与理论诊断，无需 Ollama。

```bash
bash gbhaa_experiments/launchers/run_coarse_gradient_consistency.sh
bash gbhaa_experiments/launchers/run_gb_ablation.sh
```

阶段 B：依赖 Ollama 的效率、协同与 LLM 组件。效率启动器必须先按 2.5 节预热，再串行完成 90 个子任务。

```bash
bash gbhaa_experiments/launchers/run_efficiency.sh
bash gbhaa_experiments/launchers/run_hybrid_ablation.sh
bash gbhaa_experiments/launchers/run_llm_ablation.sh
```

阶段 C：AE-PPR 与隐蔽性。

```bash
bash gbhaa_experiments/launchers/run_ae_ppr_comparison.sh
bash gbhaa_experiments/launchers/run_stealthiness.sh
```

隐蔽性实验需要保存原始和扰动后的图与特征，因此会重新运行四类攻击，不能直接用 Hybrid 普通结果代替。

只收集 GB 消融的 Citeseer 结构结果时：

```bash
DATASETS="citeseer" bash gbhaa_experiments/launchers/run_gb_ablation.sh
```

效率启动器固定为 Citeseer+Cora，不使用 `DATASETS` 覆盖。除 GB 消融和效率外，其余五个启动器均按各自协议保持 Citeseer。

<!-- DETAILS -->
