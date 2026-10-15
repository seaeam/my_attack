#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE=screen
case "${1:-}" in
  screen|summary) MODE="$1"; shift ;;
  --help|-h)
    echo 'Usage: bash gbhaa_experiments/launchers/run_post_editing.sh [screen|summary] [options]'
    echo 'Default: 4 editor factorial profiles x 3 matched arms, Citeseer seed 15, 60 actions (12 trials).'
    echo 'Preview until --execute. Use --seeds 15 16 and a new --output-root for 24 trials.'
    echo 'Summary accepts --root and --output-dir. Interrupted runs: --execute --retry-failed.'
    exit 0 ;;
esac
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then PYTHON_BIN="$CONDA_PREFIX/bin/python"; else PYTHON_BIN=python; fi
fi
export PYTHON_BIN
RESULT_ROOT="$REPO_ROOT/gbhaa_experiments/post_editing_results"
REPORT_ROOT="$REPO_ROOT/gbhaa_experiments/followup_reports/post_editing"
cd "$REPO_ROOT"
case "$MODE" in
  screen)
    exec bash "$SCRIPT_DIR/run_post_tuning.sh" --preset editing --matched-controls \
      --datasets citeseer --seeds 15 --source-budget 0.1 --max-actions 60 \
      --output-root "$RESULT_ROOT" "$@" ;;
  summary)
    exec "$PYTHON_BIN" -m gbhaa_experiments.summarize_post_tuning \
      --root "$RESULT_ROOT" --output-dir "$REPORT_ROOT" "$@" ;;
esac
