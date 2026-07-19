#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
EXECUTE="${EXECUTE:-1}"

export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b-instruct-fp16}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"

COMMON_ARGS=(
  --model Meta-Both
  --split_data normal
  --step 1
  --freeze_structure_features
  --allow_partial_vocabulary
)

CITESEER_ARGS=(
  --level 4
  --miter 60
  --lr 0.05
  --global_important_ratio 0.45
  --global_ppr_alpha 0.08
  --global_ppr_iters 120
  --global_seed_strategy degree
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
)

CORA_ARGS=(
  --level 2
  --miter 450
  --lr 0.012
  --global_important_ratio 0.25
  --global_ppr_alpha 0.08
  --global_ppr_iters 180
  --global_seed_strategy uniform
  --text_attack_max_visits 16
  --text_retries 3
  --text_budget_per_node 80
  --text_topk_ratio 0.12
  --text_ppr_alpha 0.28
  --text_ppr_iters 60
  --text_min_cluster_size 2
  --text_max_cluster_size 4
  --text_similarity_min 0.35
  --text_cdl_topk 36
  --text_cluster_attr_topk 36
  --text_max_added_words 70
)

RUN_FLAGS=()
if [[ "$EXECUTE" == "1" ]]; then
  RUN_FLAGS+=(--execute)
fi

run_dataset() {
  local dataset="$1"
  shift

  "$PYTHON_BIN" -m gbhaa_experiments.run_matrix \
    --experiment efficiency \
    --datasets "$dataset" \
    --seeds 15 16 17 18 19 \
    --ptb-rates 0.01 0.05 0.10 \
    --python "$PYTHON_BIN" \
    --llm-type gpt \
    --api-base-url "$OPENAI_BASE_URL" \
    ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
    -- "$@"
}

cd "$REPO_ROOT"
run_dataset citeseer "${COMMON_ARGS[@]}" "${CITESEER_ARGS[@]}"
run_dataset cora "${COMMON_ARGS[@]}" "${CORA_ARGS[@]}"
