"""Offline inventory of legacy evidence; never imports or executes attack code.

Only result.json files in the four historical suites are read. Missing CLI
options remain unknown: today's defaults cannot establish historical settings.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
from typing import Any

from gbhaa_experiments import PACKAGE_ROOT
from gbhaa_experiments.specs import EXPERIMENTS, SMALL_DATASETS

DEFAULT_RESULT_ROOT = PACKAGE_ROOT / "results"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "followup_reports" / "legacy"
TOLERANCE = 3e-4
SEEDS = tuple(range(15, 20))
SCOPE = {
    "ae_ppr": (("citeseer",), (0.05, 0.1)),
    "hybrid": (("citeseer",), (0.05, 0.1)),
    "gb_ablation": (("citeseer",), (0.05, 0.1)),
    "efficiency": (("citeseer", "cora"), (0.01, 0.05, 0.1)),
}
CELL_FIELDS = ("experiment", "dataset", "ptb_rate", "seed", "variant")
TRANSITIONS = {"experiment_global_transition", "experiment_local_transition"}
CONFIG_FIELDS = (
    "global_seed_strategy",
    "global_ppr_alpha",
    "global_ppr_iters",
    "global_important_ratio",
    "global_label_weight",
    "text_ppr_alpha",
    "text_ppr_iters",
    "text_topk_ratio",
    "text_attack_nodes",
    "text_attack_max_visits",
    "text_budget_per_node",
    "local_candidate_strategy",
    "local_candidate_hops",
    "freeze_structure_features",
    "model",
    "split_data",
    "epochs",
    "miter",
    "lr",
    "level",
    "step",
    "coarsen_method",
    "experiment_coarsen_method",
    "use_text_attack",
    "llm_type",
    "experiment_global_transition",
    "experiment_local_transition",
)
CAUTIONS = [
    "This is a descriptive inventory of recorded summaries, not verification of manuscript "
    "tables/figures or paper provenance; no automatic number replacement is justified.",
    "The current default is fixed to edge-first in heir.py; the explicitly named pre-flip "
    "diagnostic retains the former attribute-before-edge behavior. Historical JSON has no verifiable "
    "source snapshot, so historical execution-order provenance remains unknown. "
    "Version edge-first follow-up results separately and never pool them with legacy runs.",
    "The reviewed global operator accumulates iterates using row-normalized P @ z; "
    "it is not established as standard stationary PageRank. A uniform source is invariant "
    "under this operator (up to numerical effects); degree-source runs are not thereby invalid. "
    "Transition labels alone do not establish the historical operator.",
    "Only explicit recorded command options are extracted; omitted options are unknown, "
    "not inferred from current defaults. Matching recorded settings cannot prove matching "
    "unrecorded defaults, code, graph/split/features, completed budgets, or RNG state.",
    "Usable complete seed blocks mean all expected arms have unique, internally consistent "
    "recorded summaries. AE-PPR blocks also require matched recorded settings except the "
    "two transition flags. This does not establish causal effects or publication readiness.",
    "The four-arm factorial may have different structural trajectories; strict selector "
    "controls instead require the same candidate universe, frozen structure features, "
    "matched trajectory hashes, and aligned victim RNG/predictions. Legacy records cannot "
    "verify these requirements. Shared numeric seeds alone do not prove pairing.",
    "Efficiency comparisons require matched hardware/runtime, model/cache conditions, and "
    "completed work. Runtime metadata is retained, but recorded elapsed time alone cannot "
    "establish a speedup. Configuration groups separate recorded profiles, without pooling.",
    "Cell identity uses directory coordinates when parseable; conflicting metadata remains "
    "unresolved. Irregular paths retain metadata claims but are unusable. Missing cells have "
    "no file mapped under this rule, distinct from present but unusable cells; a misplaced "
    "record cannot be reassigned to a missing cell without resolving its provenance. "
    "runs.jsonl and stdout logs are deliberately excluded. Metric consistency tolerance "
    "is an absolute 0.0003 for rounded accuracy summaries.",
]


def _eligible(variant, dataset):
    # Match run_matrix's allowlist precedence, including efficiency Cora/node_level.
    if variant.allowed_datasets:
        return dataset in variant.allowed_datasets
    return not variant.small_only or dataset in SMALL_DATASETS


def expected_cells():
    """Explicit requested legacy matrix, independent of discovered directories."""
    return [
        dict(
            experiment=suite,
            dataset=dataset,
            ptb_rate=rate,
            seed=seed,
            variant=variant.name,
        )
        for suite, (datasets, rates) in SCOPE.items()
        for dataset in datasets
        for rate in rates
        for seed in SEEDS
        for variant in EXPERIMENTS[suite]
        if _eligible(variant, dataset)
    ]


def _secret_key(key):
    name = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return any(
        word in name
        for word in (
            "apikey",
            "token",
            "secret",
            "password",
            "passwd",
            "credential",
            "authorization",
            "privatekey",
            "accesskey",
        )
    )


def _tokens(command):
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, list) and all(isinstance(t, str) for t in command):
        return command[:]
    raise ValueError("command is not a string or token list")


def _sanitize(value):
    """Scrub secret options/keys and copies of their values before any reporting."""
    secrets = set()

    def collect(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if _secret_key(key) and isinstance(child, str) and child:
                    secrets.add(child)
                collect(child)
        elif isinstance(item, list):
            for i, child in enumerate(item):
                if isinstance(child, str) and child.startswith("--"):
                    flag, sep, val = child.partition("=")
                    if _secret_key(flag):
                        if sep and val:
                            secrets.add(val)
                        elif i + 1 < len(item) and isinstance(item[i + 1], str):
                            secrets.add(item[i + 1])
                collect(child)
        elif isinstance(item, str) and "--" in item:
            try:
                parts = shlex.split(item)
            except ValueError:
                # Never reproduce malformed command text or its exception message.
                return
            for i, part in enumerate(parts):
                flag, sep, val = part.partition("=")
                if flag.startswith("--") and _secret_key(flag):
                    if sep and val:
                        secrets.add(val)
                    elif i + 1 < len(parts):
                        secrets.add(parts[i + 1])

    collect(value)

    def scrub(item):
        if isinstance(item, dict):
            return {
                scrub(str(key)): "<redacted>" if _secret_key(key) else scrub(child)
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [scrub(child) for child in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, str):
            for secret in sorted(secrets, key=len, reverse=True):
                item = item.replace(secret, "<redacted>")
        return item

    return scrub(value)


def _scalar(value):
    if not isinstance(value, str):
        return value
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else value


def _parse_command(command):
    tokens = _tokens(command)
    options, occurrences, positionals = {}, defaultdict(list), []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            flag, sep, value = token[2:].partition("=")
            flag = flag.replace("-", "_")
            if not sep:
                value = True
                if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                    index += 1
                    value = tokens[index]
            value = _scalar(value)
            options[flag] = value
            occurrences[flag].append(value)
        else:
            positionals.append(token)
        index += 1
    return tokens, options, dict(occurrences), positionals


def _key(cell):
    return tuple(cell[k] for k in CELL_FIELDS)


def _cell(data):
    try:
        cell = {key: data[key] for key in CELL_FIELDS}
        if not all(
            isinstance(cell[k], str) for k in ("experiment", "dataset", "variant")
        ):
            return None
        if isinstance(cell["seed"], bool) or isinstance(cell["ptb_rate"], bool):
            return None
        seed, rate = float(cell["seed"]), float(cell["ptb_rate"])
        if not math.isfinite(seed) or not seed.is_integer() or not math.isfinite(rate):
            return None
        cell.update(seed=int(seed), ptb_rate=rate, dataset=cell["dataset"].lower())
        return cell
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _path_cell(path, root):
    parts = path.relative_to(root).parts
    if len(parts) < 6:
        return None
    dataset, budget, seed, variant = parts[-5:-1]
    if not budget.startswith("ptb_") or not seed.startswith("seed_"):
        return None
    return _cell(
        dict(
            experiment=parts[0],
            dataset=dataset,
            ptb_rate=budget[4:],
            seed=seed[5:],
            variant=variant,
        )
    )


def _issue(record, code, detail, severity="error"):
    record["issues"].append(dict(code=code, severity=severity, detail=detail))


def _finite(value):
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _different(left, right):
    return abs(left - right) > TOLERANCE + 1e-12


def _nonfinite_paths(value, prefix=""):
    if isinstance(value, float) and not math.isfinite(value):
        yield prefix
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _nonfinite_paths(
                child, f"{prefix}.{key}" if prefix else str(key)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _nonfinite_paths(child, f"{prefix}[{index}]")


def _metrics(record, data, spec):
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    record["metrics"] = metrics
    primary = spec.primary_metric if spec else data.get("primary_metric")
    if not isinstance(primary, str):
        _issue(
            record,
            "primary_metric_mismatch",
            "Primary metric name is missing or invalid",
        )
        primary = None
    record["primary_metric"] = primary
    record["primary_value"] = data.get("primary_value")
    required = {"clean_accuracy"}
    if isinstance(primary, str):
        required.add(primary)
        required.add(primary.replace("accuracy", "drop"))
    if primary == "combined_accuracy":
        required.add("misclassification")
    missing = sorted(k for k in required if k not in metrics or metrics[k] is None)
    if missing:
        _issue(record, "missing_metrics", missing)
    for key, value in metrics.items():
        if not _finite(value):
            _issue(record, "invalid_metric", {"metric": key, "value": value})
            if isinstance(value, str) and value.lower().lstrip("+-") in {
                "nan",
                "inf",
                "infinity",
            }:
                _issue(record, "nonfinite", {"field": "metrics." + key})
        elif (
            key.endswith("accuracy") or key == "misclassification"
        ) and not 0 <= value <= 1:
            _issue(record, "metric_out_of_range", {"metric": key, "value": value})
    combined, misclassification = metrics.get("combined_accuracy"), metrics.get(
        "misclassification"
    )
    if (
        _finite(combined)
        and _finite(misclassification)
        and _different(1 - combined, misclassification)
    ):
        _issue(
            record,
            "misclassification_disagreement",
            dict(expected=1 - combined, recorded=misclassification),
        )
    clean = metrics.get("clean_accuracy")
    for prefix in ("edge", "feature", "combined"):
        attacked, drop = metrics.get(prefix + "_accuracy"), metrics.get(
            prefix + "_drop"
        )
        if all(_finite(v) for v in (clean, attacked, drop)) and _different(
            clean - attacked, drop
        ):
            _issue(
                record,
                "drop_disagreement",
                dict(metric=prefix + "_drop", expected=clean - attacked, recorded=drop),
            )
    if primary and data.get("primary_metric") != primary:
        _issue(
            record,
            "primary_metric_mismatch",
            dict(expected=primary, recorded=data.get("primary_metric")),
        )
    value = data.get("primary_value")
    if not _finite(value):
        _issue(record, "missing_metrics", ["primary_value"])
    elif _finite(metrics.get(primary)) and _different(value, metrics[primary]):
        _issue(
            record,
            "primary_value_disagreement",
            dict(expected=metrics[primary], recorded=value),
        )


def _read_record(path, root):
    path_cell = _path_cell(path, root)
    record = dict(
        path=str(path.relative_to(root)),
        cell=path_cell,
        mapping_source="path" if path_cell else "unresolved",
        issues=[],
        config={"source": "unknown", **dict.fromkeys(CONFIG_FIELDS)},
        provenance=dict(
            source_hash=None,
            historical_order="unknown",
            manuscript_mapping="unresolved",
        ),
    )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("root is not an object")
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        _issue(record, "corrupt_json", {"error_type": type(exc).__name__})
        return record
    nonfinite = list(_nonfinite_paths(raw))
    data = _sanitize(raw)
    for field in nonfinite:
        _issue(record, "nonfinite", {"field": field})
    metadata_cell = _cell(data)
    record["metadata_cell"] = metadata_cell
    if metadata_cell is None:
        _issue(record, "cell_mapping_unresolved", "Missing or invalid cell metadata")
    elif path_cell and path_cell != metadata_cell:
        _issue(
            record,
            "cell_mapping_conflict",
            dict(path=path_cell, metadata=metadata_cell),
        )
    elif path_cell is None:
        record.update(cell=metadata_cell, mapping_source="metadata")
        _issue(
            record,
            "path_mapping_unresolved",
            "Irregular directory layout; metadata claim retained, not verified",
        )
    else:
        record["mapping_source"] = "path_and_metadata"
    record["returncode"] = data.get("returncode")
    if type(record["returncode"]) is not int or record["returncode"] != 0:
        _issue(record, "returncode", dict(recorded=record["returncode"], expected=0))
    for field in ("wall_time_seconds", "peak_rss_mib", "runtime_environment"):
        record[field] = data.get(field)
    record["provenance"]["source_hash"] = data.get("source_hash") or data.get(
        "source_sha256"
    )
    _issue(
        record,
        "historical_provenance_unknown",
        "Source/order and paper mapping are unverified",
        "warning",
    )
    try:
        tokens, options, occurrences, positionals = _parse_command(data.get("command"))
    except ValueError:
        tokens, options, occurrences, positionals = [], {}, {}, []
        _issue(
            record,
            "command_unresolved",
            "Command missing or malformed; raw text omitted",
        )
    record["command"] = tokens
    record["config"] = dict(
        source="recorded_command" if tokens else "unknown",
        **{key: options.get(key) for key in CONFIG_FIELDS},
    )
    record["config"].update(
        options=options,
        option_occurrences=occurrences,
        invocation=positionals,
        unrecorded_fields=[key for key in CONFIG_FIELDS if key not in options],
    )
    if options.get("global_seed_strategy") == "uniform":
        _issue(
            record,
            "uniform_global_invariance",
            "Uniform source is invariant under the reviewed legacy P @ z operator; historical code unverified",
            "warning",
        )
    if metadata_cell:
        for key in ("dataset", "seed", "ptb_rate"):
            if key in options and options[key] != metadata_cell[key]:
                _issue(
                    record,
                    "command_metadata_mismatch",
                    dict(field=key, command=options[key], metadata=metadata_cell[key]),
                )
    cell = record["cell"]
    spec = (
        next(
            (
                v
                for v in EXPERIMENTS.get(cell["experiment"], ())
                if v.name == cell["variant"]
            ),
            None,
        )
        if cell
        else None
    )
    if spec:
        if not positionals or spec.target not in positionals:
            _issue(record, "entrypoint_mapping_unresolved", dict(expected=spec.target))
        _, expected_options, _, _ = _parse_command(list(spec.args))
        for key, value in expected_options.items():
            if options.get(key) != value:
                code = (
                    "transition_mapping_unresolved"
                    if key in TRANSITIONS
                    else "variant_settings_mismatch"
                )
                _issue(
                    record,
                    code,
                    dict(field=key, expected=value, recorded=options.get(key)),
                )
    else:
        _issue(
            record,
            "variant_mapping_unresolved",
            "No current spec matches this recorded cell",
        )
    _metrics(record, data, spec)
    if cell and cell["experiment"] == "efficiency":
        for field in ("wall_time_seconds", "peak_rss_mib"):
            if not _finite(record[field]) or record[field] <= 0:
                _issue(record, "missing_efficiency_metrics", field)
    return record


def _settings(record, *, across_seeds=False):
    config = record["config"]
    ignored = (
        TRANSITIONS | {"dataset", "seed", "ptb_rate"} if across_seeds else TRANSITIONS
    )
    settings = {
        k: v
        for k, v in config.get("options", {}).items()
        if k not in ignored and not _secret_key(k)
    }
    settings["invocation"] = config.get("invocation")
    settings["runtime_environment"] = record.get("runtime_environment")
    return settings


def _errors(record):
    return any(issue["severity"] == "error" for issue in record["issues"])


def audit_results(result_root=DEFAULT_RESULT_ROOT):
    """Read legacy JSON and return a strict-JSON, sanitized evidence report."""
    root = Path(result_root).resolve()
    expected = expected_cells()
    records = [
        _read_record(path, root)
        for suite in SCOPE
        for path in sorted((root / suite).rglob("result.json"))
    ]
    by_cell = defaultdict(list)
    for record in records:
        if record["cell"]:
            by_cell[_key(record["cell"])].append(record)
    for group in by_cell.values():
        if len(group) > 1:
            for record in group:
                _issue(record, "duplicate_cell", [r["path"] for r in group])
    missing = [cell for cell in expected if _key(cell) not in by_cell]
    blocks = defaultdict(list)
    for cell in expected:
        blocks[_key(cell)[:-1]].append(cell)
    mismatches, complete, block_status = [], [], []
    for block_key, cells in blocks.items():
        suite, dataset, rate, seed = block_key
        block = dict(
            experiment=suite,
            dataset=dataset,
            ptb_rate=rate,
            seed=seed,
            variants=[cell["variant"] for cell in cells],
        )
        available = [record for cell in cells for record in by_cell.get(_key(cell), [])]
        settings_by_arm = {
            record["cell"]["variant"]: _settings(record) for record in available
        }
        differences = {}
        if suite == "ae_ppr" and len(settings_by_arm) > 1:
            for field in sorted(
                {key for settings in settings_by_arm.values() for key in settings}
            ):
                values = {
                    arm: settings.get(field, "<unrecorded>")
                    for arm, settings in settings_by_arm.items()
                }
                if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
                    differences[field] = values
            if differences:
                mismatch = {**block, "differences": differences}
                mismatches.append(mismatch)
                for record in available:
                    _issue(record, "factorial_settings_mismatch", sorted(differences))
        # A paired clean baseline is a necessary recorded-summary check for all suites.
        cleans = [r.get("metrics", {}).get("clean_accuracy") for r in available]
        finite_cleans = [v for v in cleans if _finite(v)]
        if finite_cleans and _different(max(finite_cleans), min(finite_cleans)):
            for record in available:
                _issue(
                    record,
                    "clean_baseline_mismatch",
                    dict(min=min(finite_cleans), max=max(finite_cleans)),
                )
        block["missing_variants"] = [
            c["variant"] for c in cells if _key(c) not in by_cell
        ]
        block["unusable_variants"] = sorted(
            {r["cell"]["variant"] for r in available if _errors(r)}
        )
        block["complete"] = all(len(by_cell.get(_key(c), [])) == 1 for c in cells)
        block["usable"] = block["complete"] and not block["unusable_variants"]
        if block["usable"]:
            profiles = {
                r["cell"]["variant"]: _settings(r, across_seeds=True) for r in available
            }
            block["configuration_group"] = hashlib.sha256(
                json.dumps(profiles, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            block["paths"] = [r["path"] for r in available]
            complete.append(block.copy())
        block_status.append(block)
    expected_keys = {_key(c) for c in expected}
    unusable = [
        {
            **c,
            "paths": [r["path"] for r in by_cell[_key(c)]],
            "issues": sorted(
                {
                    i["code"]
                    for r in by_cell[_key(c)]
                    for i in r["issues"]
                    if i["severity"] == "error"
                }
            ),
        }
        for c in expected
        if _key(c) in by_cell and any(_errors(r) for r in by_cell[_key(c)])
    ]
    unresolved = []
    for record in records:
        record["usable_record"] = not _errors(record)
        for issue in record["issues"]:
            if "mapping" in issue["code"] or issue["code"] in {
                "command_metadata_mismatch",
                "corrupt_json",
            }:
                unresolved.append(
                    dict(
                        path=record["path"],
                        reason=issue["code"],
                        detail=issue["detail"],
                    )
                )
    if records:
        for reason, detail in (
            (
                "historical_source_and_order",
                "No verified link from result JSON to an executed source snapshot/order",
            ),
            (
                "manuscript_mapping",
                "No supplied mapping from result cells to manuscript tables/figures",
            ),
            (
                "paired_artifact_provenance",
                "Graph/split/features, trajectory, aligned predictions and RNG cannot be verified from summaries",
            ),
        ):
            unresolved.append(
                dict(reason=reason, detail=detail, paths=[r["path"] for r in records])
            )
    counts = Counter(i["code"] for r in records for i in r["issues"])
    report = dict(
        schema_version=1,
        result_root=str(root),
        tolerance=TOLERANCE,
        scope={
            suite: dict(
                datasets=list(datasets),
                budgets=list(rates),
                seeds=list(SEEDS),
                variants={
                    dataset: [
                        v.name for v in EXPERIMENTS[suite] if _eligible(v, dataset)
                    ]
                    for dataset in datasets
                },
            )
            for suite, (datasets, rates) in SCOPE.items()
        },
        cautions=CAUTIONS[:],
        expected_cells=expected,
        records=records,
        missing_cells=missing,
        unusable_cells=unusable,
        unexpected_cells=[
            dict(cell=r["cell"], path=r["path"])
            for r in records
            if not r["cell"] or _key(r["cell"]) not in expected_keys
        ],
        unresolved_mapping=unresolved,
        factorial_settings_mismatches=mismatches,
        seed_blocks=block_status,
        usable_complete_seed_blocks=complete,
        summary=dict(
            records=len(records),
            expected_cells=len(expected),
            missing_cells=len(missing),
            unusable_cells=len(unusable),
            usable_records=sum(r["usable_record"] for r in records),
            complete_seed_blocks=sum(b["complete"] for b in block_status),
            usable_complete_seed_blocks=len(complete),
            factorial_settings_mismatches=len(mismatches),
            unresolved_mapping=len(unresolved),
            issue_counts=dict(sorted(counts.items())),
        ),
    )
    if not root.is_dir():
        report["cautions"].append(
            "Result root is missing; all requested cells are absent."
        )
    return _sanitize(report)


def _cell_label(cell):
    return "/".join(str(cell[k]) for k in CELL_FIELDS)


def render_markdown(report):
    """Human-readable counts plus exact cells, configuration, and provenance gaps."""
    lines = [
        "# Legacy evidence audit",
        "",
        "Read-only inventory; no experiments were executed.",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(report["summary"], indent=2),
        "```",
        "",
        "## Scope",
        "",
        "```json",
        json.dumps(report["scope"], indent=2),
        "```",
        "",
        "## Evidence limits and plan concerns",
        "",
    ]
    lines += [f"- {caution}" for caution in report["cautions"]]
    for title, key in (
        ("Exact missing cells", "missing_cells"),
        ("Present but unusable cells", "unusable_cells"),
    ):
        lines += ["", f"## {title}", ""]
        lines += [
            f"- `{_cell_label(cell)}`"
            + (" — " + ", ".join(cell["issues"]) if "issues" in cell else "")
            for cell in report[key]
        ] or ["None."]
    lines += ["", "## Usable complete seed blocks (descriptive only)", ""]
    for block in report["usable_complete_seed_blocks"]:
        lines.append(
            f"- {block['experiment']}/{block['dataset']}/{block['ptb_rate']}/seed_{block['seed']}: "
            f"{', '.join(block['variants'])}; configuration group `{block['configuration_group']}`"
        )
    if not report["usable_complete_seed_blocks"]:
        lines.append("None.")
    for title, key in (
        ("Factorial settings mismatches", "factorial_settings_mismatches"),
        ("Unexpected cells", "unexpected_cells"),
        ("Unresolved mapping", "unresolved_mapping"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "```json",
            json.dumps(report[key], indent=2, ensure_ascii=False),
            "```",
        ]
    lines += ["", "## Per-record inventory", ""]
    lines += [
        "Complete sanitized commands, configuration details and evidence are retained in `audit.json`; credential values are represented by `<redacted>`.",
        "",
        "| Record | Global source | Global alpha / iterations | Local alpha / iterations | Issues |",
        "|---|---|---|---|---|",
    ]
    for record in report["records"]:
        config = record.get("config", {})
        issues = (
            ", ".join(
                i["code"]
                for i in record["issues"]
                if i["code"] != "historical_provenance_unknown"
            )
            or "none beyond historical provenance"
        )
        lines.append(
            f"| `{record['path']}` | {config.get('global_seed_strategy')} | "
            f"{config.get('global_ppr_alpha')} / {config.get('global_ppr_iters')} | "
            f"{config.get('text_ppr_alpha')} / {config.get('text_ppr_iters')} | {issues} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_reports(report, output_dir=DEFAULT_OUTPUT_DIR):
    """Write only new derived report paths, refusing writes inside legacy results."""
    output = Path(output_dir).resolve()
    root = Path(report["result_root"]).resolve()
    if output == root or root in output.parents:
        raise ValueError("Output directory must be outside legacy result root")
    targets = [output / name for name in ("audit.json", "audit.md")]
    if any(
        path.is_symlink() or (path.exists() and path.stat().st_nlink > 1)
        for path in targets
    ):
        raise ValueError("Report paths must not be symlinks or hard links")
    safe = _sanitize(report)
    encoded = (
        json.dumps(safe, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )
    markdown = render_markdown(safe)
    output.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(encoded, encoding="utf-8")
    targets[1].write_text(markdown, encoding="utf-8")
    return tuple(targets)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", "--results-root", type=Path, default=DEFAULT_RESULT_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = audit_results(args.result_root)
    try:
        paths = write_reports(report, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], sort_keys=True))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
