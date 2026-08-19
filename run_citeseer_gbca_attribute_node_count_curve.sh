#!/usr/bin/env bash

set -euo pipefail

# GBCA curve over the number of distinct attribute-target nodes.
# The structural perturbation rate stays at 10%, and one admitted attribute
# node may be revisited at most 15 times. Only the global distinct-node budget
# changes between runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEED=15
PTB_RATE=0.10
MAX_VISITS=15
read -r -a ATTRIBUTE_NODE_COUNTS <<< "${ATTRIBUTE_NODE_COUNTS:-700 900 1100 1300 1500}"

if [[ -x /opt/miniconda3/envs/smc/bin/python ]]; then
  DEFAULT_PYTHON=/opt/miniconda3/envs/smc/bin/python
else
  DEFAULT_PYTHON="$(command -v python3 || true)"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/gbhaa_experiments/results/attribute_node_count_curve/citeseer/ptb_0.10/seed_15}"
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

if [[ "${#ATTRIBUTE_NODE_COUNTS[@]}" -eq 0 ]]; then
  echo "ATTRIBUTE_NODE_COUNTS must contain at least one non-negative integer." >&2
  exit 2
fi

previous_count=-1
for node_count in "${ATTRIBUTE_NODE_COUNTS[@]}"; do
  if [[ ! "$node_count" =~ ^[0-9]+$ ]]; then
    echo "Invalid attribute node count '$node_count'." >&2
    exit 2
  fi
  if (( node_count <= previous_count )); then
    echo "ATTRIBUTE_NODE_COUNTS must be strictly increasing." >&2
    exit 2
  fi
  previous_count=$node_count
done

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
  --ptb_rate "$PTB_RATE"
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

  --text_attack_max_visits "$MAX_VISITS"
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
result_logs = sorted(
    output_dir.glob("attribute_nodes_*.log"),
    key=lambda path: int(path.stem.removeprefix("attribute_nodes_")),
)


def last_float(pattern, text):
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def last_int(pattern, text):
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


coverage_pattern = re.compile(
    r"Attribute attack coverage:\s*"
    r"unique_nodes=(\d+),\s*"
    r"total_visits=(\d+),\s*"
    r"unique_budget=(\d+|unlimited),\s*"
    r"max_visits=(\d+),\s*"
    r"graph_nodes=(\d+)"
)

rows = []
for log_path in result_logs:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    clean = last_float(r"Clean accuracy:\s*([0-9.]+)", text)
    attacked = last_float(r"Combined accuracy:\s*([0-9.]+)", text)
    perturbations = last_int(r"Structure perturbations=(\d+)", text)
    coverage_matches = coverage_pattern.findall(text)
    requested_nodes = int(log_path.stem.removeprefix("attribute_nodes_"))
    if clean is None or attacked is None or perturbations is None or not coverage_matches:
        continue

    actual_nodes, total_visits, unique_budget, max_visits, graph_nodes = coverage_matches[-1]
    if unique_budget == "unlimited" or int(unique_budget) != requested_nodes:
        continue
    if int(max_visits) != 15:
        continue

    actual_nodes = int(actual_nodes)
    total_visits = int(total_visits)
    graph_nodes = int(graph_nodes)
    rows.append(
        {
            "dataset": "citeseer",
            "method": "GBCA",
            "seed": 15,
            "edge_perturbation_rate": 0.10,
            "edge_perturbation_rate_percent": 10.0,
            "max_visits": 15,
            "requested_unique_attribute_nodes": requested_nodes,
            "actual_unique_attribute_nodes": actual_nodes,
            "actual_attribute_node_percent": actual_nodes / graph_nodes * 100.0,
            "attribute_target_visits": total_visits,
            "graph_nodes": graph_nodes,
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

fieldnames = [
    "dataset",
    "method",
    "seed",
    "edge_perturbation_rate",
    "edge_perturbation_rate_percent",
    "max_visits",
    "requested_unique_attribute_nodes",
    "actual_unique_attribute_nodes",
    "actual_attribute_node_percent",
    "attribute_target_visits",
    "graph_nodes",
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

echo "GBCA distinct attribute-node-count experiment"
echo "  dataset: Citeseer"
echo "  target model: GCN"
echo "  seed: $SEED"
echo "  fixed edge perturbation rate: $PTB_RATE (10%)"
echo "  fixed max_visits: $MAX_VISITS"
echo "  distinct attribute-node budgets: ${ATTRIBUTE_NODE_COUNTS[*]}"
echo "  output: $OUTPUT_DIR"
echo "  lower attacked accuracy means a stronger attack"

for node_count in "${ATTRIBUTE_NODE_COUNTS[@]}"; do
  log_path="$OUTPUT_DIR/attribute_nodes_${node_count}.log"

  if [[ -s "$log_path" ]] \
    && rg -q "Combined accuracy:" "$log_path" \
    && rg -q "unique_budget=${node_count}, max_visits=${MAX_VISITS}," "$log_path" \
    && [[ "$FORCE" == "0" ]]; then
    echo "[skip] completed unique-node budget=$node_count: $log_path"
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] unique-node budget=$node_count -> $log_path"
    continue
  fi

  if pgrep -f "meta.py.*--dataset citeseer" >/dev/null 2>&1; then
    echo "Another Citeseer meta.py experiment is running; wait for it to finish." >&2
    exit 3
  fi

  if [[ -e "$log_path" ]]; then
    backup_path="$log_path.partial.$(date +%Y%m%d_%H%M%S)"
    mv "$log_path" "$backup_path"
    echo "[resume] moved existing log to $backup_path"
  fi

  echo "[run] Citeseer / GBCA / edge=10% / max_visits=15 / unique_nodes=$node_count"
  started_at=$SECONDS
  set +e
  "$PYTHON_BIN" meta.py "${COMMON_ARGS[@]}" \
    --text_attack_total_nodes "$node_count" 2>&1 | tee "$log_path"
  run_status=${PIPESTATUS[0]}
  set -e

  if [[ "$run_status" -ne 0 ]]; then
    echo "[failed] unique-node budget=$node_count exited with status $run_status; see $log_path" >&2
    exit "$run_status"
  fi

  echo "[done] unique-node budget=$node_count completed in $((SECONDS - started_at)) seconds"
  summarize_results
done

if [[ "$DRY_RUN" == "0" ]]; then
  summarize_results
  echo "All requested GBCA runs are complete. Summary: $OUTPUT_DIR/summary.csv"
else
  echo "Dry run complete. Run without DRY_RUN=1 to execute the attacks."
fi
