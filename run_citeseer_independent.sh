#!/bin/bash
# Citeseer comparison: structure and attribute attacks select targets independently.

set -euo pipefail

export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b-instruct-fp16}"

ARGS=(
  --dataset citeseer
  --model Meta-Both
  --coarsen_method gb
  --ptb_rate 0.1

  --level 4
  --step 1
  --miter 60
  --lr 0.05

  --global_important_ratio 0.45
  --global_ppr_alpha 0.08
  --global_ppr_iters 120
  --global_seed_strategy degree

  --freeze_structure_features

  --use_text_attack
  --llm_type gpt
  --openai_api_key ollama
  --api_base_url http://localhost:11434/v1

  --text_attack_max_visits 4
  --text_retries 3
  --text_budget_per_node 35
  --text_topk_ratio 0.08
  --text_ppr_alpha 0.26
  --text_ppr_iters 30
  --text_min_cluster_size 2
  --text_max_cluster_size 5

  --text_similarity_min 0.65
  --text_cdl_topk 14
  --text_cluster_attr_topk 14
  --text_max_added_words 40

  --independent_text_seed_count 2
  --independent_text_seed_strategy feature_grad
  --independent_text_seed_pool all
)

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" meta_independent.py "${ARGS[@]}"
