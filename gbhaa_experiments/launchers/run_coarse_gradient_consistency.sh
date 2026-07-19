#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
EXECUTE="${EXECUTE:-1}"
DATASET="${DATASET:-citeseer}"

ARGS=(
  --dataset "$DATASET"
  --seed 15
  --clusters 8 16 32
  --coarsen-method gb
  --epochs 200
  --miter 10
  --lr 0.01
  --global-important-ratio 0.10
  --global-ppr-alpha 0.15
  --global-ppr-iters 30
  --global-seed-strategy degree
  --topk 5
)

cd "$REPO_ROOT"
if [[ "$EXECUTE" == "1" ]]; then
  "$PYTHON_BIN" -m gbhaa_experiments.coarse_gradient_consistency "${ARGS[@]}"
else
  printf 'Dry-run: '
  printf '%q ' "$PYTHON_BIN" -m gbhaa_experiments.coarse_gradient_consistency "${ARGS[@]}"
  printf '\nSet EXECUTE=1 to run.\n'
fi

