"""Run matched seeds/budgets for efficiency and ablation experiment matrices."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.runner import (
    append_jsonl,
    command_text,
    edge_only_base_args,
    result_payload,
    run_command,
    write_json,
)
from gbhaa_experiments.specs import EXPERIMENTS, SMALL_DATASETS, Variant, get_variants


def _target_command(python: str, variant: Variant) -> List[str]:
    if variant.target_kind == "module":
        return [python, "-m", variant.target]
    return [python, variant.target]


def build_command(
    *,
    python: str,
    variant: Variant,
    dataset: str,
    seed: int,
    ptb_rate: float,
    base_args: Sequence[str],
    llm_type: str,
    api_key: str,
    api_base_url: str,
) -> List[str]:
    command = _target_command(python, variant)
    command.extend(
        base_args if variant.accepts_text_args else edge_only_base_args(base_args)
    )
    command.extend(variant.args)
    if variant.needs_text_attack:
        command.append("--use_text_attack")
        if variant.uses_external_llm:
            command.extend(
                [
                    "--llm_type",
                    llm_type,
                    "--openai_api_key",
                    api_key,
                ]
            )
            if api_base_url:
                command.extend(["--api_base_url", api_base_url])
    command.extend(
        [
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--ptb_rate",
            str(ptb_rate),
        ]
    )
    return command


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a matched GBHA experiment matrix. Commands are dry-run unless --execute is passed."
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[15])
    parser.add_argument("--ptb-rates", nargs="+", type=float, default=[0.05])
    parser.add_argument("--variants", nargs="*", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("gbhaa_experiments/results"),
    )
    parser.add_argument("--llm-type", default="gpt")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "base_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to every underlying meta*.py command.",
    )
    args = parser.parse_args(argv)
    if args.base_args and args.base_args[0] == "--":
        args.base_args = args.base_args[1:]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    variants = get_variants(args.experiment, args.variants)
    api_key = os.environ.get(args.api_key_env, "")
    if (
        args.execute
        and args.llm_type != "llama"
        and any(v.uses_external_llm for v in variants)
        and not api_key
    ):
        raise SystemExit(
            f"{args.api_key_env} is required for executable text-attack variants. "
            "For local Ollama, set it to a non-secret placeholder such as 'ollama'."
        )

    output_root = (REPO_ROOT / args.output_root).resolve()
    summary_path = output_root / args.experiment / "runs.jsonl"
    planned = 0
    failures = 0
    for dataset in args.datasets:
        for ptb_rate in args.ptb_rates:
            for seed in args.seeds:
                for variant in variants:
                    if variant.small_only and dataset not in SMALL_DATASETS:
                        print(
                            f"SKIP {dataset}/{variant.name}: node-level search is restricted to small graphs"
                        )
                        continue
                    planned += 1
                    command = build_command(
                        python=args.python,
                        variant=variant,
                        dataset=dataset,
                        seed=seed,
                        ptb_rate=ptb_rate,
                        base_args=args.base_args,
                        llm_type=args.llm_type,
                        api_key=api_key or "<set-via-env>",
                        api_base_url=args.api_base_url,
                    )
                    label = f"{dataset}/ptb_{ptb_rate:g}/seed_{seed}/{variant.name}"
                    run_dir = output_root / args.experiment / label
                    print(f"[{args.experiment}] {label}")
                    print(f"  {command_text(command)}")
                    if not args.execute:
                        continue

                    result = run_command(
                        command,
                        cwd=REPO_ROOT,
                        log_path=run_dir / "stdout.log",
                        env=os.environ.copy(),
                    )
                    payload = result_payload(
                        result,
                        metadata={
                            "experiment": args.experiment,
                            "variant": variant.name,
                            "dataset": dataset,
                            "seed": seed,
                            "ptb_rate": ptb_rate,
                            "description": variant.description,
                        },
                        primary_metric=variant.primary_metric,
                    )
                    write_json(run_dir / "result.json", payload)
                    append_jsonl(summary_path, payload)
                    if result.returncode != 0:
                        failures += 1
                        print(f"  FAILED (exit {result.returncode}); see {run_dir / 'stdout.log'}")
                    else:
                        print(
                            "  OK "
                            f"time={result.wall_time_seconds:.2f}s "
                            f"peak_rss={result.peak_rss_mib}MiB "
                            f"primary={payload['primary_value']}"
                        )

    if not args.execute:
        print(f"Dry-run complete: {planned} commands planned. Re-run with --execute.")
    else:
        print(f"Execution complete: {planned - failures}/{planned} commands exited 0.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
