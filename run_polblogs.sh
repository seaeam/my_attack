#!/bin/bash

set -euo pipefail

export OLLAMA_MODEL="llama3.2:1b-instruct-fp16"

ARGS=(
  --dataset polblogs
  --model Meta-Both                  # 伪标签覆盖所有unlabeled节点，丰富梯度信号
  --ptb_rate 0.10                    # 扰动预算

  --coarsen_method gb                # 使用粒球层次粗化
  --level 2                          # PolBlogs约1490点，保持两层避免搜索过粗
  --step 1
  --miter 320                        # 增强代理模型收敛，提升结构梯度质量
  --lr 0.012                         # 降低过激更新，提升结构搜索稳定性

  --global_important_ratio 0.38      # 扩大候选重要节点，增加跨社区可攻击面
  --global_ppr_alpha 0.08            # 偏全局扩散，适配PolBlogs社区结构
  --global_ppr_iters 120             # 充分传播全局重要性
  --global_seed_strategy degree      # 优先围绕高连接博客节点搜索

  --freeze_structure_features        # 结构搜索使用原始特征，防止文本攻击污染梯度

  --use_text_attack                  # 启用局部属性攻击
  --allow_fallback_vocabulary        # PolBlogs为身份特征，显式使用feature_i占位词
  --llm_type gpt
  --openai_api_key ollama
  --api_base_url http://localhost:11434/v1

  --text_attack_max_visits 40
  --text_retries 3                   # 增加重试，确保生成质量
  --text_budget_per_node 240         # 身份特征攻击的单节点预算
  --text_topk_ratio 0.18             # 每步覆盖更多节点，加快属性扰动扩散
  --text_ppr_alpha 0.22              # 聚焦边端点邻域，增强feature-only迁移
  --text_ppr_iters 60                # 充分传播局部PPR
  --text_min_cluster_size 1          # 允许单节点模板，提高攻击针对性
  --text_max_cluster_size 4          # 缩小簇上限，减少模板过泛化

  --text_similarity_min 0.00         # 无真实语义词表，禁用语义相似度回拉
  --text_cdl_topk 120                # 跨类混淆维度数量
  --text_cluster_attr_topk 120       # 簇属性维度数量
  --text_max_added_words 240         # 身份特征攻击的最大新增维度数
)

# python meta.py "${ARGS[@]}"
# python meta_gin.py "${ARGS[@]}"
python meta_gsage.py "${ARGS[@]}"
