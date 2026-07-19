"""Shared subprocess execution, output parsing, and result persistence."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SECRET_FLAGS = {"--openai_api_key", "--api-key", "--token"}

# meta_edge_only.py intentionally exposes no text/LLM CLI. Hybrid launchers use
# one shared argument block, so edge variants must remove both each unsupported
# option and its following value before dispatch.
EDGE_ONLY_UNSUPPORTED_OPTIONS = {
    "--freeze_structure_features": 0,
    "--use_text_attack": 0,
    "--allow_fallback_vocabulary": 0,
    "--allow_partial_vocabulary": 0,
    "--llm_type": 1,
    "--openai_api_key": 1,
    "--api_base_url": 1,
    "--llama_model_path": 1,
    "--text_attack_nodes": 1,
    "--text_attack_max_visits": 1,
    "--text_retries": 1,
    "--text_budget_per_node": 1,
    "--text_topk_ratio": 1,
    "--text_ppr_alpha": 1,
    "--text_ppr_iters": 1,
    "--local_candidate_strategy": 1,
    "--local_candidate_hops": 1,
    "--text_min_cluster_size": 1,
    "--text_max_cluster_size": 1,
    "--text_similarity_min": 1,
    "--text_cdl_topk": 1,
    "--text_cluster_attr_topk": 1,
    "--text_max_added_words": 1,
}


@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    wall_time_seconds: float
    peak_rss_mib: Optional[float]
    output: str


_METRIC_PATTERNS = {
    "clean_accuracy": re.compile(r"Clean accuracy:\s*([0-9.]+)"),
    "edge_accuracy": re.compile(r"Edge attack accuracy:\s*([0-9.]+)"),
    "feature_accuracy": re.compile(r"Feature attack accuracy:\s*([0-9.]+)"),
    "combined_accuracy": re.compile(r"Combined accuracy:\s*([0-9.]+)"),
    "misclassification": re.compile(r"Misclassification:\s*([0-9.]+)"),
}


def redact_command(command: Sequence[str]) -> List[str]:
    """Return a command safe to store in result metadata."""
    redacted: List[str] = []
    hide_next = False
    for token in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        redacted.append(token)
        if token in SECRET_FLAGS:
            hide_next = True
    return redacted


def command_text(command: Sequence[str]) -> str:
    return shlex.join(redact_command(command))


def edge_only_base_args(args: Sequence[str]) -> List[str]:
    """Remove text-only options, including their values, from shared CLI args."""
    filtered: List[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        option_name, separator, _ = token.partition("=")
        if separator and option_name in EDGE_ONLY_UNSUPPORTED_OPTIONS:
            index += 1
            continue
        arity = EDGE_ONLY_UNSUPPORTED_OPTIONS.get(token)
        if arity is None:
            filtered.append(token)
            index += 1
            continue
        if index + arity >= len(args):
            raise ValueError(f"Missing value for base option {token}")
        index += arity + 1
    return filtered


def parse_attack_output(output: str) -> Dict[str, Any]:
    """Parse the stable summary lines printed by meta*.py entry points."""
    metrics: Dict[str, Any] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        matches = pattern.findall(output)
        if matches:
            metrics[key] = float(matches[-1])

    llm_rows = re.findall(r"LLM Calls:\s*(\d+),\s*Cache Hits:\s*(\d+)", output)
    if llm_rows:
        metrics["llm_calls"] = sum(int(row[0]) for row in llm_rows)
        metrics["cache_hits"] = sum(int(row[1]) for row in llm_rows)
    external_llm_rows = re.findall(r"External LLM Calls:\s*(\d+)", output)
    if external_llm_rows:
        metrics["llm_calls"] = sum(int(value) for value in external_llm_rows)

    perturbation_rows = re.findall(r"Structure perturbations=(\d+)", output)
    if perturbation_rows:
        metrics["structure_perturbations"] = int(perturbation_rows[-1])

    if "clean_accuracy" in metrics:
        for prefix in ("edge", "feature", "combined"):
            attacked_key = f"{prefix}_accuracy"
            if attacked_key in metrics:
                metrics[f"{prefix}_drop"] = (
                    metrics["clean_accuracy"] - metrics[attacked_key]
                )
    return metrics


def _timed_command(command: Sequence[str]) -> List[str]:
    time_bin = Path("/usr/bin/time")
    if not time_bin.exists():
        return list(command)
    if platform.system() == "Darwin":
        return [str(time_bin), "-l", *command]
    return [str(time_bin), "-v", *command]


def _parse_peak_rss(output: str) -> Optional[float]:
    linux = re.findall(
        r"Maximum resident set size \(kbytes\):\s*(\d+)", output
    )
    if linux:
        return int(linux[-1]) / 1024.0

    mac = re.findall(r"(\d+)\s+maximum resident set size", output)
    if mac:
        return int(mac[-1]) / (1024.0 * 1024.0)
    return None


@lru_cache(maxsize=1)
def runtime_environment() -> Dict[str, Any]:
    """Capture enough host context to make efficiency numbers auditable."""
    details: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ollama_model": os.environ.get("OLLAMA_MODEL"),
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            query = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=5,
            )
            if query.returncode == 0:
                details["gpus"] = [
                    line.strip() for line in query.stdout.splitlines() if line.strip()
                ]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return details


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    """Run one experiment command and retain its complete combined output."""
    wrapped = _timed_command(command)
    started = time.perf_counter()
    completed = subprocess.run(
        wrapped,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
    return CommandResult(
        command=redact_command(command),
        returncode=completed.returncode,
        wall_time_seconds=elapsed,
        peak_rss_mib=_parse_peak_rss(output),
        output=output,
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def result_payload(
    result: CommandResult,
    *,
    metadata: Dict[str, Any],
    primary_metric: str,
) -> Dict[str, Any]:
    parsed = parse_attack_output(result.output)
    return {
        **metadata,
        "command": result.command,
        "returncode": result.returncode,
        "wall_time_seconds": result.wall_time_seconds,
        "peak_rss_mib": result.peak_rss_mib,
        "runtime_environment": runtime_environment(),
        "metrics": parsed,
        "primary_metric": primary_metric,
        "primary_value": parsed.get(primary_metric),
    }
