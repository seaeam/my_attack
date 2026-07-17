#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
EXECUTE="${EXECUTE:-0}"
read -r -a DATASETS <<< "${DATASETS:-citeseer}"

export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b-instruct-fp16}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"

BASE_ARGS=(
  --model Meta-Both
  --split_data normal
  --epochs 200
  --coarsen_method gb
  --level 2
  --step 1
  --miter 10
  --lr 0.01
  --global_important_ratio 0.10
  --global_ppr_alpha 0.15
  --global_ppr_iters 30
  --global_seed_strategy degree
  --freeze_structure_features
  --allow_fallback_vocabulary
  --text_attack_max_visits 1
  --text_retries 0
  --text_budget_per_node 15
  --text_topk_ratio 0.05
  --text_ppr_alpha 0.20
  --text_ppr_iters 25
  --text_min_cluster_size 2
  --text_max_cluster_size 8
  --text_similarity_min 0.85
  --text_cdl_topk 10
  --text_cluster_attr_topk 10
  --text_max_added_words 20
)

RUN_FLAGS=()
if [[ "$EXECUTE" == "1" ]]; then
  RUN_FLAGS+=(--execute)
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" -m gbhaa_experiments.run_matrix \
  --experiment ae_ppr \
  --datasets "${DATASETS[@]}" \
  --seeds 15 16 17 18 19 \
  --ptb-rates 0.05 0.10 \
  --python "$PYTHON_BIN" \
  --llm-type gpt \
  --api-base-url "$OPENAI_BASE_URL" \
  ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
  -- "${BASE_ARGS[@]}"
