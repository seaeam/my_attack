"""Dry-run-first, resumable experiment plans; legacy results are never reused."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.runner import command_text, redact_command, run_command, write_json
from gbhaa_experiments.followup_config import (
    PROTOCOL, SELECTOR_PROTOCOL, FACTORIAL_PROTOCOL, protocol_for_suite,
    factorial_operator_settings,
)

DEFAULT_OUTPUT = REPO_ROOT / "gbhaa_experiments/followup_results"
SELECTOR_OUTPUT = REPO_ROOT / "gbhaa_experiments/followup_results_pr_pre"
FACTORIAL_OUTPUT = REPO_ROOT / "gbhaa_experiments/followup_results_factorial_pre"
SUITES = ("factorial", "selectors", "ordering", "uniform_repair")


def default_output(suite):
    return {"selectors": SELECTOR_OUTPUT, "factorial": FACTORIAL_OUTPUT}.get(suite, DEFAULT_OUTPUT)


def make_jobs(suite, seeds, budgets):
    if suite not in SUITES:
        raise ValueError(suite)
    variants = []
    if suite == "factorial":
        for global_kind in ("topology", "attribute"):
            for local in ("ppr", "ae_pr"):
                variants.append(
                    (
                        f"{global_kind}_{local}",
                        global_kind,
                        local,
                        "pre",
                        "degree",
                        "gb",
                    )
                )
    elif suite == "selectors":
        variants = [
            (s, "attribute", s, "pre", "degree", "gb")
            for s in ("ae_pr", "pr", "degree", "random")
        ]
    elif suite == "ordering":
        variants = [
            (s, "attribute", "ae_pr", s, "degree", "gb") for s in ("post", "pre")
        ]
    else:
        variants = [
            (f"{method}_{source}", "attribute", "ae_pr", "post", source, method)
            for method in ("gb", "kmeans", "node_level")
            for source in ("uniform", "degree")
        ]
    jobs = []
    for budget in budgets:
        for seed in seeds:
            for variant, global_kind, selector, local_graph, source, method in variants:
                jobs.append(
                    dict(
                        suite=suite,
                        dataset="cora" if suite == "uniform_repair" else "citeseer",
                        budget=float(budget),
                        seed=int(seed),
                        variant=variant,
                        global_transition=global_kind,
                        selector=selector,
                        local_graph=local_graph,
                        global_source=source,
                        search_method=method,
                    )
                )
    return jobs


def settings(job):
    values = dict(
        model="Meta-Both",
        split_data="normal",
        epochs=200,
        step=1,
        hidden=16,
        dropout=0.5,
        level=2,
        miter=10,
        lr=0.01,
        global_important_ratio=0.1,
        global_ppr_alpha=0.15,
        global_ppr_iters=30,
        global_seed_strategy=job["global_source"],
        text_attack_max_visits=10,
        text_retries=3,
        text_budget_per_node=80,
        text_topk_ratio=0.12,
        text_ppr_alpha=0.28,
        text_ppr_iters=60,
        text_min_cluster_size=2,
        text_max_cluster_size=4,
        text_similarity_min=0.35,
        text_cdl_topk=36,
        text_cluster_attr_topk=36,
        text_max_added_words=70,
        coarsen_method="kmeans" if job["search_method"] == "kmeans" else "gb",
    )
    if job["suite"] == "uniform_repair":
        values.update(
            miter=450,
            lr=0.012,
            global_important_ratio=0.25,
            global_ppr_alpha=0.08,
            global_ppr_iters=180,
        )
    if job["search_method"] == "node_level":
        values["level"] = 1
    return values


def generation_model(api_base_url, ollama_model):
    parsed = urlsplit(api_base_url)
    local_ollama = (
        parsed.hostname in {"localhost", "127.0.0.1", "::1"} and parsed.port == 11434
    )
    return ollama_model if local_ollama else "gpt-3.5-turbo"


def build_command(job, python, run_dir, api_base_url):
    command = [
        python,
        "-m",
        "gbhaa_experiments.entrypoints.meta_followup",
        "--followup-protocol",
        protocol_for_suite(job["suite"]),
        "--followup-selector",
        job["selector"],
        "--followup-global-transition",
        job["global_transition"],
        "--followup-local-graph",
        job["local_graph"],
        "--followup-artifacts",
        str(run_dir / "artifacts"),
    ]
    for key, value in settings(job).items():
        command.extend(["--" + key, str(value)])
    command += [
        "--dataset",
        job["dataset"],
        "--seed",
        str(job["seed"]),
        "--ptb_rate",
        str(job["budget"]),
        "--freeze_structure_features",
        "--use_text_attack",
        "--llm_type",
        "gpt",
        (
            "--allow_partial_vocabulary"
            if job["suite"] == "uniform_repair"
            else "--allow_fallback_vocabulary"
        ),
        "--api_base_url",
        api_base_url,
    ]
    return command


def source_manifest():
    # Include imported project implementation, including granular-ball utilities.
    paths = list(REPO_ROOT.glob("*.py"))
    for directory in ("gbhaa_experiments", "tools"):
        paths.extend((REPO_ROOT / directory).rglob("*.py"))
    return {
        str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(paths))
        if p.is_file()
    }


def input_manifest():
    # Dataset/cache edits invalidate resume. Ignore generated Python bytecode.
    paths = []
    for name in ("Data", "bow_cache"):
        root = REPO_ROOT / name
        if root.exists():
            paths.extend(
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix not in (".pyc", ".log")
            )
    result = {}
    for p in sorted(paths):
        digest = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(p.relative_to(REPO_ROOT))] = digest.hexdigest()
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def python_environment(python):
    """Inspect the selected interpreter only when the user requests execution."""
    code = """import importlib.metadata as m, json, sys, torch
packages = {}
for name in ["numpy", "scipy", "networkx", "torch", "torch-geometric", "deeprobust", "scikit-learn", "openai", "transformers"]:
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
print(json.dumps({"python": sys.version, "executable": sys.executable, "packages": packages,
"cuda": torch.version.cuda, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))
"""
    process = subprocess.run(
        [python, "-c", code], text=True, capture_output=True, check=True
    )
    return json.loads(process.stdout.strip().splitlines()[-1])


def resumable(record, expected_fingerprint):
    return (
        record.get("fingerprint") == expected_fingerprint
        and record.get("status") == "complete"
    )


def load_completed_artifacts(run_dir, expected_protocol=PROTOCOL):
    attack = json.loads((run_dir / "artifacts/attack.json").read_text())
    evaluation = json.loads((run_dir / "artifacts/evaluation.json").read_text())
    if (not attack.get("budget_complete") or attack.get("protocol") != expected_protocol
            or evaluation.get("protocol") != expected_protocol):
        raise ValueError("Incomplete/invalid follow-up artifacts")
    if not (run_dir / "artifacts/combined_predictions.npz").is_file():
        raise ValueError("Missing target-level predictions")
    if expected_protocol == FACTORIAL_PROTOCOL:
        if attack.get("operator_settings") != factorial_operator_settings():
            raise ValueError("Invalid factorial operator_settings")
        if (attack.get("global_source") != "degree" or attack.get("local_graph") != "pre"
                or attack.get("requested_local_graph") != "pre"
                or attack.get("selector") not in ("ppr", "ae_pr")
                or attack.get("global_transition") not in ("topology", "attribute")):
            raise ValueError("Invalid factorial source/selector/graph configuration")
    result_path = run_dir / "result.json"
    if result_path.exists():
        record = json.loads(result_path.read_text())
        # Check all suites, both on resume and before marking a child complete.
        # A successful process exit does not identify the artifacts it produced.
        if record.get("status") == "complete" and (
                record.get("attack") != attack or record.get("evaluation") != evaluation):
            raise ValueError(f"{run_dir}: result differs from saved artifacts")
        for key in ("seed", "dataset", "selector", "local_graph",
                    "global_transition", "global_source"):
            if key in record and (type(attack.get(key)) is not type(record[key])
                                  or attack.get(key) != record[key]):
                raise ValueError(
                    f"{run_dir}: attack.{key} mismatch: expected {record[key]!r}, "
                    f"got {attack.get(key)!r}")
    return attack, evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[15, 16, 17, 18, 19])
    parser.add_argument("--budgets", type=float, nargs="+", default=None)
    parser.add_argument(
        "--variants", nargs="+", help="Optional named subset for a user-run pilot"
    )
    parser.add_argument("--output-root", type=Path,
                        help="Defaults to separate directories for selectors and factorial")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train/attack; omitted means plan only",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Archive and retry incomplete attempts; never replace completed results",
    )
    args = parser.parse_args(argv)
    budgets = (
        args.budgets
        if args.budgets is not None
        # else ([0.01, 0.05, 0.1] if args.suite == "uniform_repair" else [0.05, 0.1])
        # else ([0.05] if args.suite == "uniform_repair" else [0.05])
        else ([0.1] if args.suite == "uniform_repair" else [0.05])
    )
    if (
        any(not 0 < b <= 1 for b in budgets)
        or len(set(args.seeds)) != len(args.seeds)
        or len(set(budgets)) != len(budgets)
    ):
        parser.error("Use distinct seeds and distinct budgets in (0,1]")
    jobs = make_jobs(args.suite, args.seeds, budgets)
    if args.variants:
        unknown = set(args.variants) - {job["variant"] for job in jobs}
        if unknown:
            parser.error(f"Unknown variants: {sorted(unknown)}")
        jobs = [job for job in jobs if job["variant"] in args.variants]
    root = (args.output_root or default_output(args.suite)).resolve()
    protocol = protocol_for_suite(args.suite)
    legacy = (REPO_ROOT / "gbhaa_experiments/results").resolve()
    if root == legacy or legacy in root.parents:
        parser.error("Use a separate output root, outside legacy results")
    print(
        f"{'EXECUTE' if args.execute else 'PLAN ONLY'}: {args.suite}, {len(jobs)} runs; output={root}",
        flush=True,
    )
    if not args.execute:
        for job in jobs:
            path = (
                root
                / job["suite"]
                / job["dataset"]
                / f"ptb_{job['budget']:g}"
                / f"seed_{job['seed']}"
                / job["variant"]
            )
            print(
                command_text(build_command(job, args.python, path, args.api_base_url))
            )
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error(
            "Set OPENAI_API_KEY (local Ollama may use the non-secret value 'ollama')"
        )
    environment = os.environ.copy()
    environment.update(
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    environment.setdefault("OLLAMA_MODEL", "llama3.2:1b-instruct-fp16")
    print("PREPARE: checking source, input files and Python environment...", flush=True)
    sources = source_manifest()
    provenance = {
        "protocol": protocol,
        "sources": sources,
        "input_files": input_manifest(),
        "python_executable": shutil.which(args.python)
        or str(Path(args.python).resolve()),
        "python_environment": python_environment(args.python),
        "api_base_url": args.api_base_url,
        "generation_model": generation_model(
            args.api_base_url, environment["OLLAMA_MODEL"]
        ),
        "machine": platform.platform(),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "thread_count": 1,
        "generation_temperature": 0.0,
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / f"{args.suite}_plan.json",
        {"protocol": protocol, "jobs": jobs, "provenance": provenance},
    )
    for job_index, job in enumerate(jobs, start=1):
        run_dir = (
            root
            / job["suite"]
            / job["dataset"]
            / f"ptb_{job['budget']:g}"
            / f"seed_{job['seed']}"
            / job["variant"]
        )
        expected = fingerprint(
            {"job": job, "settings": settings(job), "provenance": provenance}
        )
        path = run_dir / "result.json"
        if path.exists():
            old = json.loads(path.read_text())
            if resumable(old, expected):
                load_completed_artifacts(run_dir, protocol)
                print(f"SKIP complete: {run_dir}")
                continue
            if old.get("status") == "complete" or not args.retry_failed:
                raise SystemExit(
                    f"Refusing to overwrite {path}; choose a new output root or --retry-failed for failed attempts"
                )
        if run_dir.exists():
            if (run_dir / ".running").exists():
                raise SystemExit(
                    f"Run lock exists: {run_dir}; verify no process is running before removing it"
                )
            if not args.retry_failed:
                raise SystemExit(f"Incomplete directory {run_dir}; use --retry-failed")
            run_dir.rename(
                run_dir.with_name(run_dir.name + f".failed-{time.time_ns()}")
            )
        if source_manifest() != sources:
            raise SystemExit(
                "Implementation changed during the matrix; use a new output root"
            )
        run_dir.mkdir(parents=True)
        lock = run_dir / ".running"
        with lock.open("x") as handle:
            handle.write(str(os.getpid()))
        payload = {
            **job,
            "protocol": protocol,
            "fingerprint": expected,
            "provenance": provenance,
            "settings": settings(job),
            "status": "running",
        }
        write_json(path, payload)
        try:
            command = build_command(job, args.python, run_dir, args.api_base_url)
            # Persist the dispatched command even if training is interrupted.
            payload["command"] = redact_command(command)
            write_json(path, payload)
            print(f"RUN [{job_index}/{len(jobs)}] seed={job['seed']} "
                  f"selector={job['selector']} budget={job['budget']:g} {run_dir}", flush=True)
            print(f"Live log: {run_dir / 'stdout.log'}", flush=True)
            result = run_command(
                command,
                cwd=REPO_ROOT,
                env=environment,
                log_path=run_dir / "stdout.log",
                stream_output=True,
                redact_values=[environment["OPENAI_API_KEY"]],
            )
            payload.update(
                command=result.command,
                returncode=result.returncode,
                wall_time_seconds=result.wall_time_seconds,
                peak_rss_mib=result.peak_rss_mib,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Attack exited {result.returncode}; see stdout.log")
            attack, evaluation = load_completed_artifacts(run_dir, protocol)
            if attack["generation_model"] != provenance["generation_model"]:
                raise RuntimeError(
                    "Recorded generation model differs from planned backend model"
                )
            if input_manifest() != provenance["input_files"]:
                raise RuntimeError(
                    "Dataset/vocabulary cache changed during execution; keep artifacts, choose a fresh output root with the prepared cache"
                )
            payload.update(status="complete", attack=attack, evaluation=evaluation)
            write_json(path, payload)
            print(
                f"DONE [{job_index}/{len(jobs)}] {job['variant']} "
                f"elapsed={result.wall_time_seconds:.1f}s "
                f"accuracy={evaluation['metrics']['combined_accuracy']:.4f}",
                flush=True,
            )
        except BaseException as exc:
            payload.update(status="failed", error_type=type(exc).__name__,
                           error=str(exc) or type(exc).__name__)
            write_json(path, payload)
            print(
                f"FAILED [{job_index}/{len(jobs)}] {job['variant']}; see {run_dir / 'stdout.log'}",
                flush=True,
            )
            raise
        finally:
            lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
