"""Aggregate JSONL experiment records into manuscript-friendly mean/std CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


GROUP_FIELDS = ("experiment", "dataset", "variant", "ptb_rate")


def _flatten_numeric(value, prefix="") -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = float(value)
    return flattened


def _read_records(paths: Sequence[Path]) -> List[dict]:
    records: List[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def aggregate(records: Sequence[dict]) -> List[dict]:
    # JSONL files are append-only, so rerunning an identical seed would
    # otherwise overweight it. Keep the latest successful record per
    # experiment/dataset/variant/budget/seed and exclude failed commands.
    unique_records = {}
    for record in records:
        if record.get("returncode", 0) != 0:
            continue
        identity = tuple(record.get(field) for field in GROUP_FIELDS) + (
            record.get("seed"),
        )
        unique_records[identity] = record

    grouped = defaultdict(list)
    for record in unique_records.values():
        key = tuple(record.get(field) for field in GROUP_FIELDS)
        grouped[key].append(record)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = dict(zip(GROUP_FIELDS, key))
        row["runs"] = len(group)
        numeric_by_field = defaultdict(list)
        for record in group:
            for field, value in _flatten_numeric(record).items():
                if field in {"seed", "ptb_rate", "returncode"}:
                    continue
                if math.isfinite(value):
                    numeric_by_field[field].append(value)
        for field, values in sorted(numeric_by_field.items()):
            row[f"{field}_mean"] = statistics.fmean(values)
            row[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(row)
    return output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = aggregate(_read_records(args.inputs))
    if not rows:
        raise SystemExit("No records found")
    fieldnames = sorted({field for row in rows for field in row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
