#!/bin/bash
# BlogCatalog is loaded by DeepRobust as "blogcatalog".

set -euo pipefail

export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b-instruct-fp16}"
: "${PYTHON_BIN:=python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

# BlogCatalog is much denser than PolBlogs, so keep the default perturbation
# rate modest. Override from the shell when you want a larger run, e.g.
#   PTB_RATE=0.05 TEXT_ATTACK_NODES=48 ./run_blogcatalog.sh
: "${PTB_RATE:=0.02}"
: "${TEXT_ATTACK_NODES:=32}"

ARGS=(
  --dataset blogcatalog
  --model Meta-Both
  --ptb_rate "$PTB_RATE"

  --coarsen_method gb
  --level 3
  --step 1
  --miter 260
  --lr 0.012

  --global_important_ratio 0.25
  --global_ppr_alpha 0.08
  --global_ppr_iters 140
  --global_seed_strategy degree

  --freeze_structure_features

  --use_text_attack
  --llm_type gpt
  --openai_api_key ollama
  --api_base_url http://localhost:11434/v1

  --text_attack_nodes "$TEXT_ATTACK_NODES"
  --text_attack_max_visits 8
  --text_retries 2
  --text_budget_per_node 80
  --text_topk_ratio 0.06
  --text_ppr_alpha 0.24
  --text_ppr_iters 50
  --text_min_cluster_size 2
  --text_max_cluster_size 5

  --text_similarity_min 0.00
  --text_cdl_topk 48
  --text_cluster_attr_topk 48
  --text_max_added_words 80
)

"$PYTHON_BIN" meta.py "${ARGS[@]}"
