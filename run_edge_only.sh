#!/bin/bash
# 仅边攻击（不启用文本/特征攻击）
set -euo pipefail

ARGS=(
  --dataset citeseer
  --model Meta-Both
  --ptb_rate  0.10

  --level 2
  --step 1
  --miter 10                         
  --lr 0.01

  --global_important_ratio 0.15      
  --global_ppr_alpha 1
  --global_ppr_iters 40              
  --global_seed_strategy uniform      
)

python meta_edge_only.py "${ARGS[@]}"
