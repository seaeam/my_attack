#!/usr/bin/env bash

set -euo pipefail

# Citeseer perturbation-rate curve for GBCA.
# The 0% point is the clean accuracy parsed from the first completed run, so the
# attack itself is executed only for 5%, 10%, 15%, 20%, and 25% budgets.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEED=15
PTB_RATES=(0.05 0.10 0.15 0.20 0.25)

if [[ -x /opt/miniconda3/envs/smc/bin/python ]]; then
  DEFAULT_PYTHON=/opt/miniconda3/envs/smc/bin/python
else
  DEFAULT_PYTHON="$(command -v python3 || true)"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/gbhaa_experiments/results/perturbation_curve/citeseer/seed_15}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b-instruct-fp16}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OLLAMA_HOST/v1}"

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 2
fi

if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "FORCE must be 0 or 1." >&2
  exit 2
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "No executable Python environment found. Set PYTHON_BIN explicitly." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$DRY_RUN" == "0" ]]; then
  "$PYTHON_BIN" - <<'PY'
import importlib.util

required = ("torch", "deeprobust", "openai", "httpx", "transformers")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
PY

  if ! curl --fail --silent --show-error --max-time 5 "$OLLAMA_HOST/api/tags" \
    | rg -Fq "\"name\":\"$OLLAMA_MODEL\""; then
    echo "Ollama is unavailable or model '$OLLAMA_MODEL' is not installed." >&2
    echo "Start Ollama and verify: curl $OLLAMA_HOST/api/tags" >&2
    exit 2
  fi
fi

COMMON_ARGS=(
  --dataset citeseer
  --seed "$SEED"
  --model Meta-Both
  --split_data normal
  --coarsen_method gb
  --step 1
  --freeze_structure_features

  --level 4
  --miter 60
  --lr 0.05

  --global_important_ratio 0.45
  --global_ppr_alpha 0.08
  --global_ppr_iters 120
  --global_seed_strategy degree

  --use_text_attack
  --allow_partial_vocabulary
  --llm_type gpt
  --api_base_url "$OPENAI_BASE_URL"

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

summarize_results() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import csv
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
rate_logs = sorted(output_dir.glob("ptb_*.log"))


def last_float(pattern, text):
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def last_int(pattern, text):
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


rows = []
for log_path in rate_logs:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    clean = last_float(r"Clean accuracy:\s*([0-9.]+)", text)
    attacked = last_float(r"Combined accuracy:\s*([0-9.]+)", text)
    perturbations = last_int(r"Structure perturbations=(\d+)", text)
    if clean is None or attacked is None or perturbations is None:
        continue
    rate = float(log_path.stem.removeprefix("ptb_"))
    rows.append(
        {
            "dataset": "citeseer",
            "method": "GBCA",
            "seed": 15,
            "perturbation_rate": rate,
            "perturbation_rate_percent": rate * 100.0,
            "clean_accuracy": clean,
            "attacked_accuracy": attacked,
            "clean_accuracy_percent": clean * 100.0,
            "attacked_accuracy_percent": attacked * 100.0,
            "accuracy_drop_points": (clean - attacked) * 100.0,
            "misclassification_percent": (1.0 - attacked) * 100.0,
            "structure_perturbations": perturbations,
            "source_log": log_path.name,
        }
    )

rows.sort(key=lambda row: row["perturbation_rate"])
if rows:
    clean = rows[0]["clean_accuracy"]
    rows.insert(
        0,
        {
            "dataset": "citeseer",
            "method": "GBCA",
            "seed": 15,
            "perturbation_rate": 0.0,
            "perturbation_rate_percent": 0.0,
            "clean_accuracy": clean,
            "attacked_accuracy": clean,
            "clean_accuracy_percent": clean * 100.0,
            "attacked_accuracy_percent": clean * 100.0,
            "accuracy_drop_points": 0.0,
            "misclassification_percent": (1.0 - clean) * 100.0,
            "structure_perturbations": 0,
            "source_log": rows[0]["source_log"],
        },
    )

fieldnames = [
    "dataset",
    "method",
    "seed",
    "perturbation_rate",
    "perturbation_rate_percent",
    "clean_accuracy",
    "attacked_accuracy",
    "clean_accuracy_percent",
    "attacked_accuracy_percent",
    "accuracy_drop_points",
    "misclassification_percent",
    "structure_perturbations",
    "source_log",
]
temporary_path = output_dir / "summary.csv.tmp"
summary_path = output_dir / "summary.csv"
with temporary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
temporary_path.replace(summary_path)
print(f"Updated {summary_path} with {len(rows)} curve points.")
PY
}

echo "GBCA perturbation-rate experiment"
echo "  dataset: Citeseer"
echo "  target model: GCN"
echo "  seed: $SEED"
echo "  attack rates: ${PTB_RATES[*]} (0% comes from clean accuracy)"
echo "  output: $OUTPUT_DIR"
echo "  lower attacked accuracy means a stronger attack"

for rate in "${PTB_RATES[@]}"; do
  log_path="$OUTPUT_DIR/ptb_${rate}.log"

  if [[ -s "$log_path" ]] && rg -q "Combined accuracy:" "$log_path" && [[ "$FORCE" == "0" ]]; then
    echo "[skip] completed rate $rate: $log_path"
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] rate=$rate -> $log_path"
    continue
  fi

  if [[ -e "$log_path" ]]; then
    backup_path="$log_path.partial.$(date +%Y%m%d_%H%M%S)"
    mv "$log_path" "$backup_path"
    echo "[resume] moved existing log to $backup_path"
  fi

  echo "[run] Citeseer / GBCA / seed $SEED / rate $rate"
  started_at=$SECONDS
  set +e
  "$PYTHON_BIN" meta.py "${COMMON_ARGS[@]}" --ptb_rate "$rate" 2>&1 | tee "$log_path"
  run_status=${PIPESTATUS[0]}
  set -e

  if [[ "$run_status" -ne 0 ]]; then
    echo "[failed] rate $rate exited with status $run_status; see $log_path" >&2
    exit "$run_status"
  fi

  echo "[done] rate $rate completed in $((SECONDS - started_at)) seconds"
  summarize_results
done

if [[ "$DRY_RUN" == "0" ]]; then
  summarize_results
  echo "All requested GBCA runs are complete. Summary: $OUTPUT_DIR/summary.csv"
else
  echo "Dry run complete. Run without DRY_RUN=1 to execute the attacks."
fi
