"""Executable variant definitions for the manuscript experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class Variant:
    name: str
    target: str
    target_kind: str = "script"  # script | module
    args: Tuple[str, ...] = ()
    accepts_text_args: bool = True
    primary_metric: str = "combined_accuracy"
    needs_text_attack: bool = False
    uses_external_llm: bool = False
    allowed_datasets: Tuple[str, ...] = ()
    small_only: bool = False
    description: str = ""


SMALL_DATASETS = frozenset({"citeseer"})


EXPERIMENTS: Mapping[str, Tuple[Variant, ...]] = {
    "efficiency": (
        Variant(
            "gb",
            "meta.py",
            args=("--coarsen_method", "gb"),
            primary_metric="combined_accuracy",
            needs_text_attack=True,
            uses_external_llm=True,
            allowed_datasets=("citeseer", "cora"),
            description="Complete GB-coarsened structure and LLM-attribute attack.",
        ),
        Variant(
            "kmeans",
            "meta.py",
            args=("--coarsen_method", "kmeans"),
            primary_metric="combined_accuracy",
            needs_text_attack=True,
            uses_external_llm=True,
            allowed_datasets=("citeseer", "cora"),
            description="Complete K-Means-coarsened structure and LLM-attribute attack.",
        ),
        Variant(
            "node_level",
            "meta.py",
            args=("--coarsen_method", "gb", "--level", "1"),
            primary_metric="combined_accuracy",
            needs_text_attack=True,
            uses_external_llm=True,
            allowed_datasets=("citeseer", "cora"),
            small_only=True,
            description="Complete node-level structure and LLM-attribute attack without coarsening.",
        ),
    ),
    "gb_ablation": (
        Variant(
            "gb",
            "gbhaa_experiments.entrypoints.meta_coarsen_ablation",
            target_kind="module",
            args=("--experiment_coarsen_method", "gb"),
            accepts_text_args=False,
            primary_metric="edge_accuracy",
        ),
        Variant(
            "kmeans",
            "gbhaa_experiments.entrypoints.meta_coarsen_ablation",
            target_kind="module",
            args=("--experiment_coarsen_method", "kmeans"),
            accepts_text_args=False,
            primary_metric="edge_accuracy",
        ),
        Variant(
            "random",
            "gbhaa_experiments.entrypoints.meta_coarsen_ablation",
            target_kind="module",
            args=("--experiment_coarsen_method", "random"),
            accepts_text_args=False,
            primary_metric="edge_accuracy",
        ),
        Variant(
            "node_level",
            "meta_edge_only.py",
            args=("--coarsen_method", "gb", "--level", "1"),
            accepts_text_args=False,
            primary_metric="edge_accuracy",
            small_only=True,
        ),
    ),
    "hybrid": (
        Variant(
            "edge_only",
            "meta_edge_only.py",
            accepts_text_args=False,
            primary_metric="edge_accuracy",
            description="Structure-only GBHA.",
        ),
        Variant(
            "feature_only",
            "gbhaa_experiments.entrypoints.meta_feature_only",
            target_kind="module",
            primary_metric="feature_accuracy",
            needs_text_attack=True,
            uses_external_llm=True,
            description="Independent feature-gradient attack with adjacency fixed throughout.",
        ),
        Variant(
            "parallel",
            "meta_independent.py",
            primary_metric="combined_accuracy",
            needs_text_attack=True,
            uses_external_llm=True,
            description="Independent structure and attribute selection, then joint evaluation.",
        ),
        Variant(
            "serial",
            "meta.py",
            primary_metric="combined_accuracy",
            needs_text_attack=True,
            uses_external_llm=True,
            description="Endpoint-conditioned serial GBHA.",
        ),
    ),
    "llm_ablation": (
        Variant(
            "full",
            "meta.py",
            needs_text_attack=True,
            uses_external_llm=True,
        ),
        Variant(
            "deterministic_no_llm",
            "gbhaa_experiments.entrypoints.meta_deterministic_text",
            target_kind="module",
            needs_text_attack=True,
            description="Feature-aligned deterministic placeholder control without an LLM call.",
        ),
        Variant(
            "without_discriminative_words",
            "meta.py",
            args=("--text_cdl_topk", "0"),
            needs_text_attack=True,
            uses_external_llm=True,
        ),
        Variant(
            "without_keyword_preservation",
            "meta.py",
            args=("--text_budget_per_node", "0"),
            needs_text_attack=True,
            uses_external_llm=True,
        ),
        Variant(
            "without_similarity_projection",
            "meta.py",
            args=("--text_similarity_min", "0"),
            needs_text_attack=True,
            uses_external_llm=True,
        ),
        Variant(
            "without_added_word_cap",
            "meta.py",
            args=("--text_max_added_words", "1000000"),
            needs_text_attack=True,
            uses_external_llm=True,
        ),
    ),
    "ae_ppr": (
        Variant(
            "ppr_ppr",
            "gbhaa_experiments.entrypoints.meta_ae_ppr_ablation",
            target_kind="module",
            args=(
                "--experiment_global_transition",
                "ppr",
                "--experiment_local_transition",
                "ppr",
            ),
            needs_text_attack=True,
            uses_external_llm=True,
        ),
        Variant(
            "ae_ppr_ppr",
            "gbhaa_experiments.entrypoints.meta_ae_ppr_ablation",
            target_kind="module",
            args=(
                "--experiment_global_transition",
                "ae_ppr",
                "--experiment_local_transition",
                "ppr",
            ),
            needs_text_attack=True,
            uses_external_llm=True,
        ),
        Variant(
            "ppr_ae_ppr",
            "gbhaa_experiments.entrypoints.meta_ae_ppr_ablation",
            target_kind="module",
            args=(
                "--experiment_global_transition",
                "ppr",
                "--experiment_local_transition",
                "ae_ppr",
            ),
            needs_text_attack=True,
            uses_external_llm=True,
        ),
        Variant(
            "ae_ppr_ae_ppr",
            "gbhaa_experiments.entrypoints.meta_ae_ppr_ablation",
            target_kind="module",
            args=(
                "--experiment_global_transition",
                "ae_ppr",
                "--experiment_local_transition",
                "ae_ppr",
            ),
            needs_text_attack=True,
            uses_external_llm=True,
        ),
    ),
}


def get_variants(experiment: str, selected: Sequence[str] = ()) -> Tuple[Variant, ...]:
    if experiment not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment: {experiment}")
    variants = EXPERIMENTS[experiment]
    if not selected:
        return variants
    selected_set = set(selected)
    unknown = selected_set.difference(variant.name for variant in variants)
    if unknown:
        raise KeyError(
            f"Unknown variants for {experiment}: {', '.join(sorted(unknown))}"
        )
    return tuple(variant for variant in variants if variant.name in selected_set)
