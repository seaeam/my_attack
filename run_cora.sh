#!/bin/bash

set -euo pipefail

export OLLAMA_MODEL="llama3.2:1b-instruct-fp16"

ARGS=(
  --dataset cora                 
  --model Meta-Both                  
  --ptb_rate 0.10                    # 扰动预算

  --coarsen_method gb                # Cora历史日志中gb比当前细层级配置更利于边攻击
  --level 2                          # M≈50(cora)，提高根层候选边分辨率，避免level=3过度粗化
  --step 1                           
  --miter 450                        # 保持已验证的强边攻击配置
  --lr 0.012                         # 保持当前有效步长，配合更长内层训练

  --global_important_ratio 0.25      # 回到Edge-only=0.7407那版的结构候选覆盖
  --global_ppr_alpha 0.08            # 回到已验证更强的全局PPR强度
  --global_ppr_iters 180             # 回到已验证的PPR排序稳定度
  --global_seed_strategy uniform     # degree在日志中边攻击偏弱，改回无偏全图种子

  --freeze_structure_features        # ★ 关键：结构搜索用原始特征，防止文本攻击污染梯度

  --use_text_attack                  # 启用基于 LLM 的局部文本属性攻击
  --llm_type gpt                     
  --openai_api_key ollama            
  --api_base_url http://localhost:11434/v1

  --text_attack_max_visits 16        
  --text_retries 3                   # 增加重试确保生成质量
  --text_budget_per_node 80          # 单节点关键词上限放大，增强特征扰动
  --text_topk_ratio 0.12             # 增加每步覆盖节点，强化局部文本扰动
  --text_ppr_alpha 0.28              # 局部PPR更聚焦边端点邻域
  --text_ppr_iters 60                # 局部PPR更充分
  --text_min_cluster_size 2          # 文本攻击局部簇最小大小
  --text_max_cluster_size 4          # 缩小簇上限，让文本扰动更贴近局部语义

  --text_similarity_min 0.35         # 冲0.7以下：放松相似度约束，允许更强属性偏移
  --text_cdl_topk 36                 # 更多跨类混淆词
  --text_cluster_attr_topk 36        # 更多簇属性词
  --text_max_added_words 70          # 更强注入，优先压低Feature-only
)

# python meta.py "${ARGS[@]}"
# python meta_gin.py "${ARGS[@]}"
python meta_gsage.py "${ARGS[@]}"
