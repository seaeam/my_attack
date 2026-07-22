#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
EXECUTE="${EXECUTE:-1}"
read -r -a DATASETS <<< "${DATASETS:-citeseer pubmed}"

BASE_ARGS=(
  --model Meta-Both
  --split_data normal
  --epochs 200
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

cd "$REPO_ROOT"
"$PYTHON_BIN" -m gbhaa_experiments.run_matrix \
  --experiment gb_ablation \
  --datasets "${DATASETS[@]}" \
  --seeds 15 16 17 18 19 \
  --ptb-rates 0.05 0.10 \
  --python "$PYTHON_BIN" \
  ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
  -- "${BASE_ARGS[@]}"
