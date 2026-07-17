"""Shared dataset names used by the attack entry points and launchers."""

DEEPROBUST_DATASETS = frozenset(
    {"acm", "blogcatalog", "cora_ml", "flickr", "polblogs", "uai"}
)

SMALL_EXPERIMENT_DATASETS = ("cora_ml", "acm", "polblogs")

_DATASET_ALIASES = {
    "citesser": "citeseer",
    "cora-ml": "cora_ml",
    "pol-blogs": "polblogs",
    "pol_blogs": "polblogs",
}


def normalize_dataset_name(name: str) -> str:
    """Normalize CLI spelling while preserving unsupported names for validation."""
    normalized = name.strip().lower()
    return _DATASET_ALIASES.get(normalized, normalized)
