#!/bin/bash
# 225-758-375

set -euo pipefail

export OLLAMA_MODEL="llama3.2:1b-instruct-fp16"

ARGS=(
  --dataset polblogs                 
  --model Meta-Both                  # 伪标签覆盖所有unlabeled节点，梯度信号更丰富
  --ptb_rate 0.1                    

  --level 2                          # polblogs约1222点，M≈35；保持两层避免搜索过粗
  --step 1                           
  --miter 220                        # 增强代理模型收敛，提升结构梯度质量
  --lr 0.015                         # 略增大学习率，提高meta搜索强度

  --global_important_ratio 0.30      # 扩大候选重要节点，增加跨社区可攻击面
  --global_ppr_alpha 0.10            # 更偏全局扩散，适配polblogs社区结构
  --global_ppr_iters 60              # 更充分传播重要性
  --global_seed_strategy degree      # 优先围绕高连接博客节点搜索，强化结构破坏

  --freeze_structure_features        # ★ 关键：结构搜索用原始特征，防止文本攻击污染梯度

  --use_text_attack                  # 启用基于 LLM 的局部文本属性攻击
  --llm_type gpt                     
  --openai_api_key ollama            
  --api_base_url http://localhost:11434/v1

  --text_attack_max_visits 10        
  --text_retries 3                   # 增加重试确保生成质量
  --text_budget_per_node 120         # polblogs是合成feature词表，放大单节点注入维度
  --text_topk_ratio 0.10             # 每步覆盖10%节点，让特征扰动更快扩散
  --text_ppr_alpha 0.15              # 比0.20更全局，避免局部邻域反复重访
  --text_ppr_iters 45                # 局部PPR更充分
  --text_min_cluster_size 1          # 允许单节点模板，提高特征攻击针对性
  --text_max_cluster_size 4          # 缩小簇上限，减少模板过泛化

  --text_similarity_min 0.20         # polblogs无真实语义词表，放宽投影约束以增强特征破坏
  --text_cdl_topk 60                 # 更多跨类混淆维度
  --text_cluster_attr_topk 60        # 更多簇属性维度
  --text_max_added_words 120         # 允许更多feature_i维度注入
)

python meta.py "${ARGS[@]}"
