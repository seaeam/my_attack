#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE=screen
case "${1:-}" in
  screen|summary) MODE="$1"; shift ;;
  --help|-h)
    echo 'Usage: bash gbhaa_experiments/launchers/run_post_transfer.sh [screen|summary] [options]'
    echo 'Default: 5 profiles x 3 matched arms, Citeseer seed 15, 60 actions (15 trials).'
    echo 'Profiles: edit_control, edit_both, transfer_relu, transfer_residual, transfer_both.'
    echo 'Preview until --execute. Two-seed check: --seeds 15 16 and a new --output-root (30 trials).'
    echo 'Interrupted runs: --execute --retry-failed. Summary accepts --root and --output-dir.'
    exit 0 ;;
esac
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then PYTHON_BIN="$CONDA_PREFIX/bin/python"; else PYTHON_BIN=python; fi
fi
export PYTHON_BIN
RESULT_ROOT="$REPO_ROOT/gbhaa_experiments/post_transfer_results"
REPORT_ROOT="$REPO_ROOT/gbhaa_experiments/followup_reports/post_transfer"
cd "$REPO_ROOT"
case "$MODE" in
  screen)
    exec bash "$SCRIPT_DIR/run_post_tuning.sh" --preset transfer --matched-controls \
      --datasets citeseer --seeds 15 --source-budget 0.1 --max-actions 60 \
      --output-root "$RESULT_ROOT" "$@" ;;
  summary)
    exec "$PYTHON_BIN" -m gbhaa_experiments.summarize_post_tuning \
      --root "$RESULT_ROOT" --output-dir "$REPORT_ROOT" "$@" ;;
esac
