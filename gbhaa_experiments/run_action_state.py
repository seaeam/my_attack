"""Plan-only by default; run and resume isolated shared-action experiment blocks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.action_state_protocol import PROTOCOL, ARMS, DATASETS, make_jobs, protocol_settings
from gbhaa_experiments.run_followup import (
    fingerprint, source_manifest, input_manifest, python_environment, generation_model,
)
from gbhaa_experiments.runner import command_text, redact_command, run_command

DEFAULT_OUTPUT = REPO_ROOT / "gbhaa_experiments/followup_results_action_state"


def job_directory(root, job):
    return root/job["dataset"]/f"ptb_{job['budget']:g}"/f"seed_{job['seed']}"/job["variant"]


def build_command(job, python, root, api_base_url, cache):
    directory=job_directory(root,job)
    trajectory=directory.parent/"structure/artifacts/trajectory.json"
    command=[python,"-u","-m","gbhaa_experiments.entrypoints.meta_action_state",
        "--action-state-phase",job["variant"],"--action-state-artifacts",str(directory/"artifacts"),
        "--action-state-trajectory",str(trajectory),"--action-state-cache",str(cache)]
    for key,value in protocol_settings().items():
        command.extend(["--"+key,str(value)])
    command += ["--dataset",job["dataset"],"--seed",str(job["seed"]),
                "--ptb_rate",str(job["budget"]),"--freeze_structure_features"]
    if job["variant"]!="structure":
        command += ["--use_text_attack","--llm_type","gpt","--allow_partial_vocabulary",
                    "--api_base_url",api_base_url]
    return command


def can_resume(record, expected):
    return record.get("status")=="complete" and record.get("fingerprint")==expected


def check_unchanged_inputs(sources, inputs):
    if source_manifest()!=sources:
        raise ValueError("Project source changed during this experiment; use a new output root")
    if input_manifest()!=inputs:
        raise ValueError("Project input files changed during this experiment; use a new output root")


def _write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        pending=Path(f.name)
    os.replace(pending,path)


@contextmanager
def _lock(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/".execution.lock").open("a+") as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another action-state runner is using this output root") from exc
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def _execute(args,jobs,root):
    # Heavy imports/artifact reads only after an explicit --execute.
    from gbhaa_experiments.action_state_validation import load_stage, validate_result
    environment=os.environ.copy()
    environment.update(PYTHONHASHSEED="0",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    environment.setdefault("OLLAMA_MODEL","llama3.2:1b-instruct-fp16")
    print("Preparing source/input/environment fingerprints...",flush=True)
    sources=source_manifest()
    provenance={"protocol":PROTOCOL,"sources":sources,"input_files":input_manifest(),
        "python_executable":shutil.which(args.python) or str(Path(args.python).resolve()),
        "python_environment":python_environment(args.python),"api_base_url":args.api_base_url,
        "generation_model":generation_model(args.api_base_url,environment["OLLAMA_MODEL"]),
        "machine":platform.platform(),"cuda_visible_devices":environment.get("CUDA_VISIBLE_DEVICES"),
        "thread_count":1,"generation_temperature":0.}
    cache=root/"_generation_cache"/fingerprint(provenance)
    _write_json(root/"plan.json",{"protocol":PROTOCOL,"jobs":jobs,"provenance":provenance})
    for index,job in enumerate(jobs,1):
        check_unchanged_inputs(sources, provenance["input_files"])
        directory=job_directory(root,job);path=directory/"result.json"
        trajectory_hash=None
        if job["variant"]!="structure":
            plan_dir=directory.parent/"structure"
            plan=validate_result(plan_dir/"result.json",{**job,"variant":"structure"})
            if plan["provenance"]!=provenance:
                raise ValueError("Shared structural plan has different source/input/environment provenance")
            trajectory_hash=plan["trajectory"]["actions_hash"]
        expected=fingerprint({"job":job,"settings":protocol_settings(),"provenance":provenance,
                              "trajectory_hash":trajectory_hash})
        if path.exists():
            old=json.loads(path.read_text())
            if can_resume(old,expected):
                validate_result(path,job)
                print(f"SKIP verified complete [{index}/{len(jobs)}]: {directory}",flush=True)
                continue
            if old.get("status")=="complete":
                raise ValueError(f"Completed result has different configuration; use a new --output-root: {directory}")
            if not args.retry_failed:
                raise ValueError(f"Incomplete attempt; use --retry-failed to archive and retry: {directory}")
            archive=directory.with_name(directory.name+".failed-"+str(time.time_ns()))
            shutil.move(str(directory),str(archive))
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Untracked artifacts exist; use a new --output-root: {directory}")
        command=build_command(job,args.python,root,args.api_base_url,cache)
        record={**job,"protocol":PROTOCOL,"settings":protocol_settings(),"provenance":provenance,
                "fingerprint":expected,"status":"running","command":redact_command(command)}
        _write_json(path,record)
        try:
            print(f"RUN [{index}/{len(jobs)}] {job['dataset']} {job['budget']:g} seed={job['seed']} {job['variant']}",flush=True)
            result=run_command(command,cwd=REPO_ROOT,log_path=directory/"stdout.log",env=environment)
            record.update(returncode=result.returncode,wall_time_seconds=result.wall_time_seconds,
                          peak_rss_mib=result.peak_rss_mib)
            if result.returncode:
                raise RuntimeError(f"Child exited {result.returncode}; see {directory/'stdout.log'}")
            check_unchanged_inputs(sources, provenance["input_files"])
            record.update(load_stage(directory,job))
            record["status"]="complete"
            _write_json(path,record)
            validate_result(path,job)
        except BaseException as exc:
            record.update(status="failed",error_type=type(exc).__name__,error=str(exc))
            _write_json(path,record)
            print(f"FAILED {directory}: {exc}",file=sys.stderr,flush=True)
            raise
        print(f"COMPLETE [{index}/{len(jobs)}] {directory}",flush=True)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=["citeseer","cora"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[15,16,17,18,19])
    parser.add_argument("--budgets",nargs="+",type=float,default=[.05,.1])
    parser.add_argument("--python",default=sys.executable)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--api-base-url",default=os.environ.get("OPENAI_BASE_URL","http://localhost:11434/v1"))
    parser.add_argument("--execute",action="store_true",help="Run training and generation; default only prints commands")
    parser.add_argument("--retry-failed",action="store_true",help="Archive incomplete attempts and retry")
    args=parser.parse_args(argv)
    try:
        jobs=make_jobs(args.datasets,args.seeds,args.budgets)
    except ValueError as exc:
        parser.error(str(exc))
    root=args.output_root.resolve()
    forbidden=[REPO_ROOT/"gbhaa_experiments"/name for name in (
        "results","followup_results","followup_results_pr_pre","followup_results_factorial_pre")]
    if any(root==p.resolve() or p.resolve() in root.parents for p in forbidden):
        parser.error("Use a separate output root outside all legacy experiment directories")
    blocks=len(jobs)//4
    print(f"{'EXECUTE' if args.execute else 'PLAN ONLY'}: {blocks} shared trajectories + {3*blocks} arms; output={root}",flush=True)
    if not args.execute:
        for job in jobs:
            print(command_text(build_command(job,args.python,root,args.api_base_url,root/"_generation_cache/PROVENANCE")))
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required (local Ollama: ollama)")
    with _lock(root):
        return _execute(args,jobs,root)


if __name__=="__main__":raise SystemExit(main())
