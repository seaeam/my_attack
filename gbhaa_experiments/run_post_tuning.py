"""Small validation-only Endpoint-Post development sweep; plan-only by default."""
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

from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.post_tuning_protocol import (
    PROTOCOL, DATASETS, get_profile, make_jobs, profile_catalog, runtime_settings,
)
from gbhaa_experiments.run_action_state import _lock, _write_json, check_unchanged_inputs
from gbhaa_experiments.run_followup import (
    fingerprint, source_manifest, input_manifest, python_environment, generation_model,
)
from gbhaa_experiments.runner import command_text, redact_command, run_command

DEFAULT_OUTPUT = REPO_ROOT / "gbhaa_experiments/post_tuning_results"
DEFAULT_PARENT = REPO_ROOT / "gbhaa_experiments/followup_results_action_state"


def job_directory(root, job):
    return (root / job["dataset"] / f"source_ptb_{job['source_budget']:g}" /
            f"steps_{job['max_actions']}" / f"seed_{job['seed']}" / job["trial_id"])


def parent_directory(root, job):
    return root / job["dataset"] / f"ptb_{job['source_budget']:g}" / f"seed_{job['seed']}" / "structure"


def build_command(job, python, root, parent_root, api_base_url, cache):
    command = [python, "-u", "-m", "gbhaa_experiments.entrypoints.meta_post_tuning",
        "--tune-profile", job["profile_id"], "--tune-arm", job["arm"],
        "--tune-artifacts", str(job_directory(root, job)/"artifacts"),
        "--tune-source-run", str(parent_directory(parent_root, job)),
        "--tune-cache", str(cache), "--tune-max-actions", str(job["max_actions"])]
    for key, value in runtime_settings(job["config"]).items():
        command.extend(["--"+key, str(value)])
    command.extend(["--dataset", job["dataset"], "--seed", str(job["seed"]),
        "--ptb_rate", str(job["source_budget"]), "--freeze_structure_features",
        "--use_text_attack", "--llm_type", "gpt", "--allow_partial_vocabulary",
        "--api_base_url", api_base_url])
    return command


def _recommendation(args):
    """Select a saved development winner; a later matched run uses other seeds."""
    if args.profiles:
        raise ValueError("--profiles and --recommendation are mutually exclusive")
    path = args.recommendation.resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if (data.get("protocol") != PROTOCOL or data.get("status") != "complete" or
            data.get("selection_split") != "validation"):
        raise ValueError("Recommendation must be complete and selected on validation only")
    entries = data.get("recommendations")
    if not isinstance(entries, list):
        raise ValueError("Missing recommendation entries")
    chosen = set(); screening_seeds = set()
    for dataset in args.datasets:
        matches = [item for item in entries if item.get("dataset") == dataset and
                   item.get("source_budget") == args.source_budget]
        if len(matches) != 1:
            raise ValueError(f"Expected one recommendation for {dataset} at this source budget")
        item = matches[0]
        if item.get("config") != get_profile(item["profile_id"]):
            raise ValueError("Recommendation configuration differs from the catalog")
        seeds = item.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds):
            raise ValueError("Recommendation must record screening seeds")
        screening_seeds.update(seeds); chosen.add(item["profile_id"])
    if len(chosen) != 1:
        raise ValueError("Datasets selected different profiles; run each dataset separately")
    if screening_seeds.intersection(args.seeds):
        raise ValueError("Use seeds distinct from the screening seeds for this matched recheck")
    args.profiles = list(chosen)
    args.matched_controls = True
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "screening_seeds": sorted(screening_seeds)}


def _check_selection(selection):
    if selection and hashlib.sha256(Path(selection["path"]).read_bytes()).hexdigest() != selection["sha256"]:
        raise ValueError("Recommendation changed during execution; use a new output root")


def check_python_runtime(python):
    """Import required generation code before persisting any experiment plan."""
    process = subprocess.run(
        [python, "-c", "from text_attack_generator import TextAttackGenerator"],
        cwd=REPO_ROOT, text=True, capture_output=True, timeout=60)
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip() or "import failed"
        raise RuntimeError(
            f"Text-attack dependency preflight failed for {python}. "
            "Select the prepared environment with PYTHON_BIN or --python; "
            f"no experiment plan has been written.\n{detail}")


def _plan_differences(saved, current, prefix=""):
    if isinstance(saved, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(set(saved) | set(current)):
            name = f"{prefix}.{key}" if prefix else key
            if key not in saved or key not in current:
                differences.append(name)
            else:
                differences.extend(_plan_differences(saved[key], current[key], name))
        return differences
    return [prefix or "plan"] if saved != current else []


def _plan_mismatch_message(saved, current):
    differences = _plan_differences(saved, current)
    details = ", ".join(differences[:12])
    if len(differences) > 12:
        details += f", ... ({len(differences)} changed fields)"
    old_provenance = saved.get("provenance", {}) if isinstance(saved, dict) else {}
    old_python = old_provenance.get("python_executable") if isinstance(old_provenance, dict) else None
    new_python = current["provenance"]["python_executable"]
    return (f"Saved plan differs: {details}.\n"
            f"Saved Python: {old_python}\nCurrent Python: {new_python}\n"
            "Use a new --output-root for this environment/configuration. "
            "Existing plans/results are preserved; --retry-failed does not override a plan mismatch.")


def _execute(args, jobs, root, selection=None):
    # Artifact reads, environment probes, and child processes require --execute.
    from gbhaa_experiments.action_state_validation import validate_result as validate_parent
    from gbhaa_experiments.post_tuning_validation import load_trial, validate_result
    check_python_runtime(args.python)
    parents = {}
    for job in jobs:
        directory = parent_directory(args.parent_root, job)
        if directory not in parents:
            parent_job = dict(dataset=job["dataset"], seed=job["seed"],
                              budget=job["source_budget"], variant="structure")
            parents[directory] = validate_parent(directory/"result.json", parent_job)
        if len(parents[directory]["trajectory"]["actions"]) < job["max_actions"]:
            raise ValueError(f"Requested prefix exceeds saved structural trajectory: {directory}")
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    environment.setdefault("OLLAMA_MODEL", "llama3.2:1b-instruct-fp16")
    print("Preparing source/input/environment fingerprints...", flush=True)
    sources = source_manifest()
    provenance = {"protocol": PROTOCOL, "sources": sources, "input_files": input_manifest(),
        "python_executable": shutil.which(args.python) or str(Path(args.python).resolve()),
        "python_environment": python_environment(args.python), "api_base_url": args.api_base_url,
        "generation_model": generation_model(args.api_base_url, environment["OLLAMA_MODEL"]),
        "machine": platform.platform(), "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "thread_count": 1, "generation_temperature": 0.}
    if selection:
        provenance["selection_record"] = selection
    cache = root/"_generation_cache"/fingerprint(provenance)
    plan = {"protocol": PROTOCOL, "jobs": jobs, "provenance": provenance,
            "parent_root": str(args.parent_root)}
    plan_path = root/"plan.json"
    if plan_path.exists():
        saved_plan = json.loads(plan_path.read_text())
        if saved_plan != plan:
            raise ValueError(_plan_mismatch_message(saved_plan, plan))
    _write_json(plan_path, plan)
    for index, job in enumerate(jobs, 1):
        check_unchanged_inputs(sources, provenance["input_files"])
        _check_selection(selection)
        directory = job_directory(root, job); path = directory/"result.json"
        parent_run = parent_directory(args.parent_root, job)
        parent_fingerprint = parents[parent_run]["fingerprint"]
        expected = fingerprint({"job": job, "provenance": provenance,
                                "parent_fingerprint": parent_fingerprint})
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("status") == "complete":
                if old.get("fingerprint") != expected:
                    raise ValueError(f"Completed result differs; use a new --output-root: {directory}")
                validate_result(path, job)
                print(f"SKIP verified complete [{index}/{len(jobs)}]: {directory}", flush=True)
                continue
            if not args.retry_failed:
                raise ValueError(f"Incomplete attempt; use --retry-failed to archive and retry: {directory}")
            archive = directory.with_name(directory.name+".failed-"+str(time.time_ns()))
            shutil.move(str(directory), str(archive))
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Untracked artifacts exist; use a new --output-root: {directory}")
        command = build_command(job, args.python, root, args.parent_root, args.api_base_url, cache)
        record = {**job, "protocol": PROTOCOL, "settings": runtime_settings(job["config"]),
            "provenance": provenance, "parent_run": str(parent_run),
            "parent_fingerprint": parent_fingerprint, "fingerprint": expected,
            "status": "running", "command": redact_command(command)}
        _write_json(path, record)
        try:
            print(f"RUN [{index}/{len(jobs)}] {job['dataset']} seed={job['seed']} {job['trial_id']}", flush=True)
            result = run_command(command, cwd=REPO_ROOT, log_path=directory/"stdout.log", env=environment)
            record.update(returncode=result.returncode, wall_time_seconds=result.wall_time_seconds,
                          peak_rss_mib=result.peak_rss_mib)
            if result.returncode:
                raise RuntimeError(f"Child exited {result.returncode}; see {directory/'stdout.log'}")
            check_unchanged_inputs(sources, provenance["input_files"])
            _check_selection(selection)
            record.update(load_trial(directory, job, parent_run))
            record["status"] = "complete"
            _write_json(path, record)
            validate_result(path, job)
        except BaseException as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            _write_json(path, record)
            print(f"FAILED {directory}: {exc}", file=sys.stderr, flush=True)
            raise
        print(f"COMPLETE [{index}/{len(jobs)}] {directory}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["citeseer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[15])
    parser.add_argument("--source-budget", type=float, default=.1)
    parser.add_argument("--max-actions", type=int, default=60)
    parser.add_argument("--preset", choices=("tiny", "quick", "extended", "refine", "mechanism", "coverage", "editing", "transfer", "objective"), default="quick")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--matched-controls", action="store_true")
    parser.add_argument("--recommendation", type=Path)
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--execute", action="store_true", help="Run the trials; default only prints commands")
    parser.add_argument("--retry-failed", action="store_true", help="Archive incomplete attempts and retry")
    args = parser.parse_args(argv)
    if args.list_profiles:
        for profile_id, config in profile_catalog().items():
            print(f"{profile_id}: {config['description']}")
        return 0
    try:
        selection = _recommendation(args) if args.recommendation else None
        jobs = make_jobs(args.datasets, args.seeds, args.source_budget, args.max_actions,
                         args.preset, args.profiles, args.matched_controls)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    root = args.output_root.resolve(); args.parent_root = args.parent_root.resolve()
    forbidden = [REPO_ROOT/"gbhaa_experiments"/name for name in (
        "results", "followup_results", "followup_results_pr_pre", "followup_results_factorial_pre",
        "followup_results_action_state")]
    forbidden.append(args.parent_root)
    if any(root == p.resolve() or p.resolve() in root.parents or root in p.resolve().parents for p in forbidden):
        parser.error("Use a separate output root outside all legacy and parent experiment directories")
    print(f"{'EXECUTE' if args.execute else 'PLAN ONLY'}: {len(jobs)} short trials, "
          f"{args.max_actions} actions each; output={root}", flush=True)
    if not args.execute:
        for job in jobs:
            print(command_text(build_command(job, args.python, root, args.parent_root,
                                             args.api_base_url, root/"_generation_cache/PROVENANCE")))
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required (local Ollama: ollama)")
    with _lock(root):
        return _execute(args, jobs, root, selection)


if __name__ == "__main__":
    raise SystemExit(main())
