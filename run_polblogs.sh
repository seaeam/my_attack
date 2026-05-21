#!/bin/bash
# 225-758-375

set -euo pipefail

export OLLAMA_MODEL="llama3.2:1b-instruct-fp16"

ARGS=(
  --dataset polblogs                 
  --model Meta-Both                  # 伪标签覆盖所有unlabeled节点，梯度信号更丰富
  --ptb_rate 0.1                    

  --level 2                          # polblogs约1490点，M≈38；保持两层避免搜索过粗
  --step 1                           
  --miter 320                        # 增强代理模型收敛，提升结构梯度质量
  --lr 0.012                         # 降低过激更新，提升结构搜索稳定性

  --global_important_ratio 0.38      # 扩大候选重要节点，增加跨社区可攻击面
  --global_ppr_alpha 0.08            # 更偏全局扩散，适配polblogs社区结构
  --global_ppr_iters 120             # 更充分传播重要性
  --global_seed_strategy degree      # 优先围绕高连接博客节点搜索，强化结构破坏

  --freeze_structure_features        # ★ 关键：结构搜索用原始特征，防止文本攻击污染梯度

  --use_text_attack                  # 启用基于 LLM 的局部文本属性攻击
  --llm_type gpt                     
  --openai_api_key ollama            
  --api_base_url http://localhost:11434/v1

  --text_attack_max_visits 40       
  --text_retries 3                   # 增加重试确保生成质量
  --text_budget_per_node 240         # fallback特征直接替换为对立伪标签身份维度
  --text_topk_ratio 0.18             # 每步覆盖更多节点，让属性扰动更快扩散
  --text_ppr_alpha 0.22              # 更聚焦边端点邻域，增强feature-only迁移
  --text_ppr_iters 60                # 局部PPR更充分
  --text_min_cluster_size 1          # 允许单节点模板，提高特征攻击针对性
  --text_max_cluster_size 4          # 缩小簇上限，减少模板过泛化

  --text_similarity_min 0.00         # polblogs无真实语义词表，禁用语义相似度回拉
  --text_cdl_topk 120                # 更多跨类混淆维度
  --text_cluster_attr_topk 120       # 更多簇属性维度
  --text_max_added_words 240         # 直接身份特征攻击使用该上限
)

python meta.py "${ARGS[@]}"
