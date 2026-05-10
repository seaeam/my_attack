#!/bin/bash

set -euo pipefail

export OLLAMA_MODEL="llama3.2:1b-instruct-fp16"

ARGS=(
  --dataset polblogs                 
  --model Meta-Both                  # 伪标签覆盖所有unlabeled节点，梯度信号更丰富
  --ptb_rate 0.10                    # 扰动预算

  --level 2                          # M≈52(cora)，粗化图梯度精度高
  --step 1                           
  --miter 150                        # 代理模型充分收敛
  --lr 0.01                           # 更小学习率，训练更稳定

  --global_important_ratio 0.15      # 精简种子集，搜索更分散
  --global_ppr_alpha 0.15            # 更全局化，覆盖跨社区弱连接
  --global_ppr_iters 40              # 充分收敛但保留局部性（150过度收敛→退化为度中心性）
  --global_seed_strategy uniform     # 均匀初始化

  --freeze_structure_features        # ★ 关键：结构搜索用原始特征，防止文本攻击污染梯度

  --use_text_attack                  # 启用基于 LLM 的局部文本属性攻击
  --llm_type gpt                     
  --openai_api_key ollama            
  --api_base_url http://localhost:11434/v1

  --text_attack_max_visits 10        
  --text_retries 3                   # 增加重试确保生成质量
  --text_budget_per_node 50          # 单节点关键词上限放大
  --text_topk_ratio 0.05             # 每步覆盖5%节点，扩散更均匀（0.15太激进，几步就全覆盖）
  --text_ppr_alpha 0.20              # 局部PPR扩散更广
  --text_ppr_iters 35                # 局部PPR更充分
  --text_min_cluster_size 2          # 文本攻击局部簇最小大小
  --text_max_cluster_size 6          # 稍放大簇上限

  --text_similarity_min 0.55         # 保持与原始特征合理相似度（0.35太低→特征偏移过大）
  --text_cdl_topk 20                 # 更多跨类混淆词
  --text_cluster_attr_topk 20        # 更多簇属性词
  --text_max_added_words 25          # 注入词数合理（70/500词汇=14%太激进）
)

python meta.py "${ARGS[@]}"
