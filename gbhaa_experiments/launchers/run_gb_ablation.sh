#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
EXECUTE="${EXECUTE:-0}"
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
