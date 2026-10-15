#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE=screen
case "${1:-}" in
  screen|confirm|summary-screen|summary-confirm) MODE="$1"; shift ;;
  --help|-h)
    echo "Usage: bash gbhaa_experiments/launchers/run_post_refinement.sh [screen|confirm|summary-screen|summary-confirm] [runner options]"
    echo "Default screen: 10 matched profiles, seed 15, 60 actions; preview until --execute."
    echo "Confirm: saved screen recommendation, seeds 16-19, 120 actions; preview until --execute."
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
SCREEN_ROOT="$REPO_ROOT/gbhaa_experiments/post_refinement_results"
CONFIRM_ROOT="$REPO_ROOT/gbhaa_experiments/post_refinement_confirm_results"
SCREEN_REPORT="$REPO_ROOT/gbhaa_experiments/followup_reports/post_refinement"
CONFIRM_REPORT="$REPO_ROOT/gbhaa_experiments/followup_reports/post_refinement_confirm"
cd "$REPO_ROOT"
case "$MODE" in
  screen)
    exec bash "$SCRIPT_DIR/run_post_tuning.sh" --preset refine --matched-controls \
      --datasets citeseer --seeds 15 --source-budget 0.1 --max-actions 60 \
      --output-root "$SCREEN_ROOT" "$@" ;;
  confirm)
    exec bash "$SCRIPT_DIR/run_post_tuning.sh" --recommendation "$SCREEN_REPORT/recommendation.json" \
      --datasets citeseer --seeds 16 17 18 19 --source-budget 0.1 --max-actions 120 \
      --output-root "$CONFIRM_ROOT" "$@" ;;
  summary-screen)
    exec "$PYTHON_BIN" -m gbhaa_experiments.summarize_post_tuning \
      --root "$SCREEN_ROOT" --output-dir "$SCREEN_REPORT" "$@" ;;
  summary-confirm)
    exec "$PYTHON_BIN" -m gbhaa_experiments.summarize_post_tuning \
      --root "$CONFIRM_ROOT" --output-dir "$CONFIRM_REPORT" "$@" ;;
esac
