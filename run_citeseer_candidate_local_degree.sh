#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

export PYTHONUNBUFFERED=1
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b-instruct-fp16}"

python meta.py \
  --dataset citeseer \
  --model Meta-Both \
  --coarsen_method gb \
  --ptb_rate 0.1 \
  --level 4 \
  --step 1 \
  --miter 60 \
  --lr 0.05 \
  --global_important_ratio 0.45 \
  --global_ppr_alpha 0.08 \
  --global_ppr_iters 120 \
  --global_seed_strategy degree \
  --freeze_structure_features \
  --use_text_attack \
  --llm_type gpt \
  --openai_api_key ollama \
  --api_base_url http://localhost:11434/v1 \
  --text_attack_nodes 32 \
  --text_attack_max_visits 4 \
  --text_retries 3 \
  --text_budget_per_node 35 \
  --text_topk_ratio 0.08 \
  --text_ppr_alpha 0.26 \
  --text_ppr_iters 30 \
  --local_candidate_strategy local_degree \
  --local_candidate_hops 2 \
  --text_min_cluster_size 2 \
  --text_max_cluster_size 5 \
  --text_similarity_min 0.65 \
  --text_cdl_topk 14 \
  --text_cluster_attr_topk 14 \
  --text_max_added_words 40 \
  2>&1 | tee "logs/citeseer_candidate_local_degree_$(date +%Y-%m-%d_%H-%M-%S).txt"
