#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  else
    PYTHON_BIN=python
  fi
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b-instruct-fp16}"
if [[ "$OPENAI_BASE_URL" == "http://localhost:11434/v1" || "$OPENAI_BASE_URL" == "http://127.0.0.1:11434/v1" ]]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
fi
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m gbhaa_experiments.run_post_tuning --python "$PYTHON_BIN" "$@"
