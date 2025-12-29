# Meta.py 运行示例

## 基本使用

### 1. 传统 Heir 攻击（不使用文本生成）

```bash
# Citeseer数据集，5%扰动率
python meta.py --dataset citeseer --ptb_rate 0.05

# Cora数据集，10%扰动率
python meta.py --dataset cora --ptb_rate 0.10

# 使用不同的层次聚类层数
python meta.py --dataset citeseer --ptb_rate 0.05 --level 3
```

### 2. 使用文本攻击（GPT）

```bash
# 使用OpenAI GPT进行文本攻击
python meta.py \
    --dataset cora \
    --ptb_rate 0.05 \
    --use_text_attack \
    --llm_type gpt \
    --openai_api_key YOUR_API_KEY

# 或者使用环境变量
export OPENAI_API_KEY="your-api-key-here"
python meta.py --dataset cora --ptb_rate 0.05 --use_text_attack --llm_type gpt --openai_api_key $OPENAI_API_KEY
```

### 3. 使用文本攻击（本地 Llama）

```bash
# 使用本地Llama模型
python meta.py \
    --dataset cora \
    --ptb_rate 0.05 \
    --use_text_attack \
    --llm_type llama \
    --llama_model_path meta-llama/Llama-3-8b
```

## 完整参数说明

### 原有参数

| 参数         | 类型  | 默认值    | 说明                                                            |
| ------------ | ----- | --------- | --------------------------------------------------------------- |
| `--dataset`  | str   | citeseer  | 数据集名称 (cora/citeseer/pubmed/computers/photo/cs/physics 等) |
| `--ptb_rate` | float | 0.05      | 扰动率（修改的边数占总边数的比例）                              |
| `--model`    | str   | Meta-Both | 攻击模式: Meta-Both/Meta-Self/Meta-Train                        |
| `--seed`     | int   | 15        | 随机种子                                                        |
| `--epochs`   | int   | 200       | 训练轮数                                                        |
| `--lr`       | float | 0.01      | 学习率                                                          |
| `--hidden`   | int   | 16        | 隐藏层维度                                                      |
| `--level`    | int   | 2         | 层次聚类的层数                                                  |
| `--step`     | int   | 1         | 每次迭代的扰动步数                                              |
| `--miter`    | int   | 10        | 元学习迭代次数                                                  |
| `--oracle`   | flag  | False     | 是否使用 oracle 模式                                            |

### 新增文本攻击参数

| 参数                 | 类型 | 默认值 | 说明                                |
| -------------------- | ---- | ------ | ----------------------------------- |
| `--use_text_attack`  | flag | False  | 启用文本生成攻击节点特征            |
| `--llm_type`         | str  | gpt    | LLM 类型: gpt 或 llama              |
| `--openai_api_key`   | str  | None   | OpenAI API 密钥（使用 GPT 时必需）  |
| `--llama_model_path` | str  | None   | Llama 模型路径（使用 Llama 时必需） |

## 使用场景示例

### 场景 1: 快速测试（不使用文本攻击）

```bash
python meta.py --dataset cora --ptb_rate 0.05
```

**预期输出**:

- Clean graph accuracy: ~81%
- Edge-only attack: ~75%
- Feature-only attack: ~81% (无变化，因为没有文本攻击)
- Combined attack: ~75%

### 场景 2: 完整攻击（边+文本特征）

```bash
python meta.py \
    --dataset cora \
    --ptb_rate 0.05 \
    --use_text_attack \
    --llm_type gpt \
    --openai_api_key YOUR_KEY
```

**预期输出**:

- Clean graph accuracy: ~81%
- Edge-only attack: ~75%
- Feature-only attack: ~70% (文本攻击生效)
- Combined attack: ~60% (边+特征双重攻击)

### 场景 3: 大规模攻击

```bash
python meta.py \
    --dataset citeseer \
    --ptb_rate 0.10 \
    --use_text_attack \
    --llm_type gpt \
    --openai_api_key YOUR_KEY \
    --level 3 \
    --step 20
```

### 场景 4: 不同数据集测试

```bash
# Pubmed
python meta.py --dataset pubmed --ptb_rate 0.05 --use_text_attack --llm_type gpt --openai_api_key YOUR_KEY

# Computers (Amazon)
python meta.py --dataset computers --ptb_rate 0.05 --use_text_attack --llm_type gpt --openai_api_key YOUR_KEY
```

## 输出示例

```
📝 Text Attack Enabled
============================================================
  LLM Type: gpt
  OpenAI API Key: ********************...
  Attack Target: Node Features (via text generation)
============================================================

✅ Text attack generator initialized with LLM: gpt

🚀 Starting Heir Attack...
  Perturbation budget: 270 edges
  Perturbation rate: 5.0%
  Attack mode: Meta-Both
  Text attack: Enabled (LLM: gpt)

Perturbing graph: 100%|████████████| 14/14 [01:23<00:00,  5.96s/it]

🔥 Attacking 270 nodes with text generation (budget=20)
  ✓ Processed 10/270 nodes
  ✓ Processed 20/270 nodes
  ...
✅ Text attack completed on 270 important nodes

============================================================
📊 Evaluation Results
============================================================

=== Clean graph ===
Clean graph results: loss= 0.5432 accuracy= 0.8152

=== Edge-only attack ===
Edge-only attack results: loss= 0.7821 accuracy= 0.7543

=== Feature-only attack ===
Feature-only attack results: loss= 0.8234 accuracy= 0.7012

=== Combined attack (edge + feature) ===
Combined attack results: loss= 1.0234 accuracy= 0.6123

============================================================
📈 Attack Performance Summary
============================================================
  Clean accuracy:         0.8152
  Edge attack accuracy:   0.7543 (drop: 0.0609)
  Feature attack accuracy: 0.7012 (drop: 0.1140)
  Combined accuracy:      0.6123 (drop: 0.2029)
============================================================
```

## 故障排除

### 问题 1: "TextAttackGenerator not available"

**解决**: 确保`text_attack_generator.py`在同一目录下

### 问题 2: "BoW vocabulary file not found"

**解决**: 从 Text-level-Graph-Attack-master 复制词汇表文件：

```bash
mkdir -p bow_cache
cp Text-level-Graph-Attack-master/bow_cache/cora_vocab.pkl bow_cache/
cp Text-level-Graph-Attack-master/bow_cache/citeseer_vocab.pkl bow_cache/
cp Text-level-Graph-Attack-master/bow_cache/pubmed_vocab.pkl bow_cache/
```

### 问题 3: OpenAI API 错误

**解决**:

- 检查 API 密钥是否正确
- 检查账户余额
- 使用`--llm_type llama`切换到本地模型

### 问题 4: 内存不足

**解决**:

- 降低`--ptb_rate`
- 减少文本攻击的节点数量
- 使用较小的数据集

## 注意事项

1. **文本攻击需要 BoW 特征**: 确保数据集使用 Bag-of-Words 特征表示
2. **API 成本**: 使用 GPT 会产生 API 调用费用，建议先小规模测试
3. **计算时间**: 文本生成较慢，大规模攻击可能需要数小时
4. **GPU 内存**: 使用 Llama 需要至少 16GB GPU 显存

## 快速开始

最简单的使用方式（不需要 API 密钥）：

```bash
# 传统攻击（无需额外配置）
python meta.py --dataset cora --ptb_rate 0.05
```

使用文本攻击（需要 API 密钥）：

```bash
# 设置环境变量
export OPENAI_API_KEY="your-key"

# 运行
python meta.py --dataset cora --ptb_rate 0.05 --use_text_attack --llm_type gpt --openai_api_key $OPENAI_API_KEY
```
