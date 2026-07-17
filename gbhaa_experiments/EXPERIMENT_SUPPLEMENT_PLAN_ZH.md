# GBHA 完整补充实验方案与运行手册

本文档基于当前 `gbhaa_experiments` 中的实际代码，目的是把七项补充实验落实为可以执行、检查、汇总并写入论文的完整流程。代码目录说明见 [`README.md`](./README.md)。

本文档回答四个问题：每项实验验证什么、怎样保持公平对照、当前代码如何运行、什么样的结果才可用于论文。

## 一、实验总目标

| 论文主张 | 对应实验 | 必要证据 |
|---|---|---|
| Granular-Ball 粗化降低搜索代价并保留有效方向 | 大规模效率、GB 消融、粗粒度梯度一致性 | 时间/内存、攻击效果、粗细梯度排序一致性 |
| 结构与属性协同优于单一或完全独立攻击 | Hybrid Coupling 消融 | `serial` 与 `edge_only`、`feature_only`、`parallel` 的配对结果 |
| LLM 及其约束组件各自有效 | LLM 消融 | 完整方法、无 LLM 和各组件移除版本的比较 |
| AE-PPR 与约束改善传播及隐蔽性 | AE-PPR 2×2、隐蔽性分析 | 全局/局部贡献和效果—隐蔽性权衡 |

不要预先假定 GB、`serial` 或完整 LLM 版本一定最好，也不要只保留符合预期的随机种子。

## 二、统一实验协议

### 2.1 数据集

所有小图实验统一使用 `citeseer`，不使用其他小图数据集。

| 实验 | 数据集 |
|---|---|
| 大规模效率 | Citeseer、Pubmed、CS |
| GB 消融 | Citeseer、Pubmed |
| Hybrid、LLM、梯度一致性、AE-PPR、隐蔽性 | Citeseer |

Pubmed 和 CS 用于检验结构搜索随图规模增加时的时间与内存表现，不作为额外小图文本实验。

### 2.2 种子与预算

正式矩阵统一使用种子 `15, 16, 17, 18, 19`。

- 大规模效率：1%、5%、10%；
- 其余带预算实验：5%、10%；
- 梯度一致性：比较 8、16、32 个粒球，不执行完整攻击预算。

同一张表中的方法必须使用相同数据划分、种子和预算。

### 2.3 固定参数

| 参数 | 当前正式值 |
|---|---:|
| 攻击模型 | `Meta-Both` |
| 数据划分 | `normal` |
| 训练轮数 | 200 |
| 层次深度 | 2 |
| 每步扰动 | 1 |
| 内部迭代 | 10 |
| 学习率 | 0.01 |
| 全局重要节点比例 | 0.10 |
| 全局 PPR 重启概率/迭代 | 0.15 / 30 |
| 全局种子策略 | `degree` |
| 结构搜索特征 | 固定为原始特征 |

属性实验额外固定：每节点保留特征词 15、局部候选比例 0.05、局部 PPR 为 0.20/25、属性簇大小 2 到 8、最低余弦相似度 0.85、类别判别特征数 10、簇属性特征数 10、最大新增特征数 20、单节点最多访问 1 次、LLM 重试 0 次。

消融实验只改变其名称对应因素，其他条件保持一致。

### 2.4 Citeseer 特征词表边界

Citeseer 特征维数为 3,703，但仓库现有缓存词表只有 500 个词。启动器通过 `--allow_fallback_vocabulary` 显式使用 3,703 个 `feature_i` 占位词。

因此，相关实验只能表述为“LLM 辅助的离散特征空间扰动”，不能表述为原始论文文本改写、真实单词攻击、自然语言语义保持或文本流畅性验证。

### 2.5 统计规则

当前汇总代码计算五个种子的均值和样本标准差，正式表格统一报告 `mean ± std`，并同时给出数据集、预算、变体、有效种子数、Clean Accuracy、Attacked Accuracy 和 Accuracy Drop。

当前代码不自动计算显著性检验。没有另行完成配对统计检验时，不要声称“具有统计显著性”。

## 三、完整矩阵规模

| 实验 | 攻击子任务 | 需要外部 LLM 的子任务 |
|---|---:|---:|
| 大规模效率 | 105 | 0 |
| GB 消融 | 70 | 0 |
| Hybrid 消融 | 40 | 30 |
| LLM 消融 | 60 | 50 |
| AE-PPR 对比 | 40 | 40 |
| 隐蔽性 | 40 | 30 |
| 粗粒度梯度一致性 | 1 个诊断进程，内部含 3 种粒球数 | 0 |

完整方案为 355 个攻击子任务和 1 个梯度诊断，其中 150 个攻击子任务会访问 OpenAI 兼容接口。一次子任务可能产生多次 LLM 请求。

不要并发执行七个正式启动器。效率实验必须尽量串行，否则时间和内存不可比。

## 四、环境准备

所有命令从仓库根目录执行：

```bash
cd /Users/bytedance/Downloads/Github/my_attack
export PYTHON_BIN=/opt/miniconda3/bin/python
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
du -sh Data bow_cache
```

Citeseer 已位于 `Data/citeseer.npz`。Pubmed 可由 DeepRobust 加载路径获取，CS 由 PyG Coauthor 数据加载器准备。首次运行前需保证可访问数据源或已有缓存。

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
"$PYTHON_BIN" -m unittest tests.test_text_attack_generator -v
"$PYTHON_BIN" -m unittest discover -s gbhaa_experiments/tests -v
"$PYTHON_BIN" -m compileall -q \
  gbhaa_experiments text_attack_generator.py tests/test_text_attack_generator.py
bash -n gbhaa_experiments/launchers/*.sh
```

预期：4 项词表测试和 15 项实验框架测试通过，编译及 Shell 检查无错误。

预览所有命令，默认不会真正训练：

```bash
for launcher in gbhaa_experiments/launchers/*.sh; do
  echo "===== $launcher ====="
  bash "$launcher"
done
```

确认小图文本实验均为 `--dataset citeseer`，且包含 `--allow_fallback_vocabulary`。

## 六、低成本烟测

烟测只检查流程，不作为论文结果。

### 6.1 无 LLM 结构烟测

```bash
"$PYTHON_BIN" -m gbhaa_experiments.run_matrix \
  --experiment efficiency --datasets citeseer \
  --seeds 15 --ptb-rates 0.001 --variants gb \
  --python "$PYTHON_BIN" \
  --output-root gbhaa_experiments/smoke_results \
  --execute -- \
  --model Meta-Both --split_data normal --epochs 5 \
  --coarsen_method gb --level 2 --step 1 --miter 1 --lr 0.01 \
  --global_important_ratio 0.10 --global_ppr_alpha 0.15 \
  --global_ppr_iters 5 --global_seed_strategy degree
```

通过条件：退出码为 0，生成 `result.json` 与 `stdout.log`，并含 Clean Accuracy 和 Edge attack accuracy。

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
OPENAI_BASE_URL=http://localhost:11434/v1 \
OLLAMA_MODEL=llama3.2:1b-instruct-fp16 \
"$PYTHON_BIN" -m gbhaa_experiments.run_matrix \
  --experiment llm_ablation --datasets citeseer \
  --seeds 15 --ptb-rates 0.001 --variants full \
  --python "$PYTHON_BIN" --llm-type gpt \
  --api-base-url "$OPENAI_BASE_URL" \
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

若外部 LLM 变体的 `metrics.llm_calls` 为 0，先检查候选节点、生成错误和跳过日志，不要直接接受该结果。

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
EXECUTE=1 PYTHON_BIN="$PYTHON_BIN" bash gbhaa_experiments/launchers/run_coarse_gradient_consistency.sh
EXECUTE=1 PYTHON_BIN="$PYTHON_BIN" bash gbhaa_experiments/launchers/run_gb_ablation.sh
EXECUTE=1 PYTHON_BIN="$PYTHON_BIN" bash gbhaa_experiments/launchers/run_efficiency.sh
```

阶段 B：协同与 LLM 组件。

```bash
EXECUTE=1 PYTHON_BIN="$PYTHON_BIN" bash gbhaa_experiments/launchers/run_hybrid_ablation.sh
EXECUTE=1 PYTHON_BIN="$PYTHON_BIN" bash gbhaa_experiments/launchers/run_llm_ablation.sh
```

阶段 C：AE-PPR 与隐蔽性。

```bash
EXECUTE=1 PYTHON_BIN="$PYTHON_BIN" bash gbhaa_experiments/launchers/run_ae_ppr_comparison.sh
EXECUTE=1 PYTHON_BIN="$PYTHON_BIN" bash gbhaa_experiments/launchers/run_stealthiness.sh
```

隐蔽性实验需要保存原始和扰动后的图与特征，因此会重新运行四类攻击，不能直接用 Hybrid 普通结果代替。

只收集 Citeseer 结构结果时：

```bash
DATASETS="citeseer" EXECUTE=1 PYTHON_BIN="$PYTHON_BIN" bash gbhaa_experiments/launchers/run_gb_ablation.sh
DATASETS="citeseer" EXECUTE=1 PYTHON_BIN="$PYTHON_BIN" bash gbhaa_experiments/launchers/run_efficiency.sh
```

其他五个启动器已默认使用 Citeseer。

<!-- DETAILS -->
