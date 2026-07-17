"""Generate artifacts and compute matched stealthiness metrics for attack variants."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.runner import (
    append_jsonl,
    command_text,
    edge_only_base_args,
    result_payload,
    run_command,
    write_json,
)
from gbhaa_experiments.stealthiness import analyze_artifact_dir


VARIANTS = {
    "edge_only": ("edge", "edge"),
    "feature_only": ("feature", "feature"),
    "parallel": ("independent", "combined"),
    "serial": ("serial", "combined"),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[15])
    parser.add_argument("--ptb-rates", nargs="+", type=float, default=[0.05])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--llm-type", default="gpt")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("gbhaa_experiments/results/stealthiness"),
    )
    parser.add_argument("--skip-transitivity", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("base_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.base_args and args.base_args[0] == "--":
        args.base_args = args.base_args[1:]
    return args


def _command(
    args: argparse.Namespace,
    *,
    base: str,
    dataset: str,
    seed: int,
    ptb_rate: float,
    artifact_dir: Path,
    api_key: str,
) -> List[str]:
    base_args = edge_only_base_args(args.base_args) if base == "edge" else args.base_args
    command = [
        args.python,
        "-m",
        "gbhaa_experiments.entrypoints.meta_artifact",
        "--experiment_base",
        base,
        "--experiment_artifact_dir",
        str(artifact_dir),
        *base_args,
    ]
    if base != "edge":
        command.extend(
            [
                "--use_text_attack",
                "--llm_type",
                args.llm_type,
                "--openai_api_key",
                api_key,
            ]
        )
        if args.api_base_url:
            command.extend(["--api_base_url", args.api_base_url])
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "")
    needs_text = any(VARIANTS[name][0] != "edge" for name in args.variants)
    if args.execute and needs_text and not api_key:
        raise SystemExit(f"{args.api_key_env} is required for text attack variants")
    output_root = (REPO_ROOT / args.output_root).resolve()
    failures = 0
    planned = 0
    for dataset in args.datasets:
        for ptb_rate in args.ptb_rates:
            for seed in args.seeds:
                by_base: Dict[str, List[Tuple[str, str]]] = {}
                for variant in args.variants:
                    base, mode = VARIANTS[variant]
                    by_base.setdefault(base, []).append((variant, mode))
                for base, analyses in by_base.items():
                    planned += 1
                    run_dir = output_root / dataset / f"ptb_{ptb_rate:g}" / f"seed_{seed}" / base
                    artifact_dir = run_dir / "artifacts"
                    command = _command(
                        args,
                        base=base,
                        dataset=dataset,
                        seed=seed,
                        ptb_rate=ptb_rate,
                        artifact_dir=artifact_dir,
                        api_key=api_key or "<set-via-env>",
                    )
                    print(f"[stealthiness] {dataset}/ptb_{ptb_rate:g}/seed_{seed}/{base}")
                    print(f"  {command_text(command)}")
                    if not args.execute:
                        continue
                    result = run_command(
                        command,
                        cwd=REPO_ROOT,
                        log_path=run_dir / "stdout.log",
                        env=os.environ.copy(),
                    )
                    run_payload = result_payload(
                        result,
                        metadata={
                            "experiment": "stealthiness",
                            "base": base,
                            "dataset": dataset,
                            "seed": seed,
                            "ptb_rate": ptb_rate,
                        },
                        primary_metric="combined_accuracy" if base != "edge" else "edge_accuracy",
                    )
                    write_json(run_dir / "run_result.json", run_payload)
                    if result.returncode != 0:
                        failures += 1
                        continue
                    for variant, mode in analyses:
                        analysis = analyze_artifact_dir(
                            artifact_dir,
                            attack_mode=mode,
                            include_transitivity=not args.skip_transitivity,
                        )
                        payload = {
                            "experiment": "stealthiness",
                            "variant": variant,
                            "dataset": dataset,
                            "seed": seed,
                            "ptb_rate": ptb_rate,
                            "attack_metrics": run_payload["metrics"],
                            **analysis,
                        }
                        variant_dir = run_dir.parent / variant
                        write_json(variant_dir / "stealthiness.json", payload)
                        append_jsonl(output_root / "runs.jsonl", payload)
    if not args.execute:
        print(f"Dry-run complete: {planned} attack commands planned. Re-run with --execute.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
