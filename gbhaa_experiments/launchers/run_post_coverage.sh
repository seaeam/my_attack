#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE=screen
case "${1:-}" in
  screen|summary) MODE="$1"; shift ;;
  --help|-h)
    echo "Usage: bash gbhaa_experiments/launchers/run_post_coverage.sh [screen|summary] [options]"
    echo "Default: baseline + original editgain + decay + fresh3, 3 matched arms, Citeseer seeds 15/16, 60 actions (24 trials)."
    echo "Preview until --execute; --seeds 15 gives 12 trials. Summary accepts --root and --output-dir."
    exit 0 ;;
esac
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  else
    PYTHON_BIN=python
  fi
fi
export PYTHON_BIN
RESULT_ROOT="$REPO_ROOT/gbhaa_experiments/post_coverage_results"
REPORT_ROOT="$REPO_ROOT/gbhaa_experiments/followup_reports/post_coverage"
cd "$REPO_ROOT"
case "$MODE" in
  screen)
    exec bash "$SCRIPT_DIR/run_post_tuning.sh" --preset coverage --matched-controls \
      --datasets citeseer --seeds 15 16 --source-budget 0.1 --max-actions 60 \
      --output-root "$RESULT_ROOT" "$@" ;;
  summary)
    exec "$PYTHON_BIN" -m gbhaa_experiments.summarize_post_tuning \
      --root "$RESULT_ROOT" --output-dir "$REPORT_ROOT" "$@" ;;
esac
