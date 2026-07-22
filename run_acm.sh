#!/bin/bash

set -euo pipefail

export OLLAMA_MODEL="llama3.2:1b-instruct-fp16"

ARGS=(
  --dataset acm
  --model Meta-Both
  --ptb_rate 0.05                    # ACM使用较小扰动预算

  --coarsen_method gb                # 使用粒球层次粗化
  --level 2                          # 保持两层搜索，避免候选边过度粗化
  --step 1
  --miter 320                        # 代理模型内层训练轮数
  --lr 0.012                         # 结构搜索步长

  --global_important_ratio 0.25      # 全局重要节点覆盖比例
  --global_ppr_alpha 0.08            # 全局PPR扩散强度
  --global_ppr_iters 140             # 全局PPR迭代次数
  --global_seed_strategy uniform     # 使用无偏全图种子

  --freeze_structure_features        # 结构搜索使用原始特征，防止文本攻击污染梯度

  --use_text_attack                  # 启用局部属性攻击
  --allow_fallback_vocabulary        # ACM无对齐词表，显式使用feature_i占位词
  --llm_type gpt
  --openai_api_key ollama
  --api_base_url http://localhost:11434/v1

  --text_attack_max_visits 16
  --text_retries 3                   # 增加重试，确保生成质量
  --text_budget_per_node 64          # 单节点属性预算
  --text_topk_ratio 0.10             # 每步属性攻击节点覆盖比例
  --text_ppr_alpha 0.26              # 局部PPR聚焦边端点邻域
  --text_ppr_iters 50                # 局部PPR迭代次数
  --text_min_cluster_size 2          # 局部簇最小大小
  --text_max_cluster_size 5          # 局部簇最大大小

  --text_similarity_min 0.35         # 属性相似度下限
  --text_cdl_topk 32                 # 跨类混淆属性数量
  --text_cluster_attr_topk 32        # 簇属性数量
  --text_max_added_words 64          # 最大新增属性数
)

# python meta.py "${ARGS[@]}"
# python meta_gin.py "${ARGS[@]}"
python meta_gsage.py "${ARGS[@]}"
