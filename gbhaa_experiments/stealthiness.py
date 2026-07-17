"""Structural and feature-space stealthiness metrics for saved attack artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from gbhaa_experiments.runner import write_json


def _binary_adjacency(value: sp.spmatrix) -> sp.csr_matrix:
    result = value.tocsr().astype(np.float64)
    result.eliminate_zeros()
    result.data[:] = 1.0
    return result


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    length = max(len(left), len(right))
    p = np.pad(left.astype(float), (0, length - len(left)))
    q = np.pad(right.astype(float), (0, length - len(right)))
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    midpoint = 0.5 * (p + q)

    def kl(source, target):
        mask = source > 0
        return float(np.sum(source[mask] * np.log(source[mask] / target[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def _homophily(adj: sp.spmatrix, labels: np.ndarray) -> float:
    upper = sp.triu(_binary_adjacency(adj), k=1).tocoo()
    if upper.nnz == 0:
        return 0.0
    return float(np.mean(labels[upper.row] == labels[upper.col]))


def _transitivity(adj: sp.spmatrix) -> float:
    binary = _binary_adjacency(adj).tolil()
    binary.setdiag(0)
    binary = binary.tocsr()
    degrees = np.asarray(binary.sum(axis=1)).reshape(-1)
    triples_twice = float(np.sum(degrees * np.maximum(degrees - 1.0, 0.0)))
    if triples_twice == 0:
        return 0.0
    closed_walks = float((binary @ binary).multiply(binary).sum())
    return closed_walks / triples_twice


def structural_metrics(
    original: sp.spmatrix,
    modified: sp.spmatrix,
    labels: np.ndarray,
    *,
    include_transitivity: bool = True,
) -> Dict[str, float]:
    original_binary = _binary_adjacency(original)
    modified_binary = _binary_adjacency(modified)
    original_diag = float(original_binary.diagonal().sum())
    modified_diag = float(modified_binary.diagonal().sum())
    original_no_diag = original_binary.copy().tolil()
    modified_no_diag = modified_binary.copy().tolil()
    original_no_diag.setdiag(0)
    modified_no_diag.setdiag(0)
    original_no_diag = original_no_diag.tocsr()
    modified_no_diag = modified_no_diag.tocsr()
    original_no_diag.eliminate_zeros()
    modified_no_diag.eliminate_zeros()

    difference = (modified_no_diag != original_no_diag).astype(np.int8)
    undirected_flips = int(sp.triu(difference, k=1).nnz)
    original_edges = int(sp.triu(original_no_diag, k=1).nnz)
    modified_edges = int(sp.triu(modified_no_diag, k=1).nnz)
    original_degree = np.asarray(original_no_diag.sum(axis=1)).reshape(-1).astype(int)
    modified_degree = np.asarray(modified_no_diag.sum(axis=1)).reshape(-1).astype(int)
    original_hist = np.bincount(original_degree)
    modified_hist = np.bincount(modified_degree)
    original_components = connected_components(original_no_diag, directed=False)[0]
    modified_components = connected_components(modified_no_diag, directed=False)[0]
    original_homophily = _homophily(original_no_diag, labels)
    modified_homophily = _homophily(modified_no_diag, labels)
    metrics: Dict[str, float] = {
        "original_edges": float(original_edges),
        "modified_edges": float(modified_edges),
        "edge_flips": float(undirected_flips),
        "edge_flip_rate": float(undirected_flips / max(original_edges, 1)),
        "original_self_loops": original_diag,
        "modified_self_loops": modified_diag,
        "asymmetry_entries": float((modified_no_diag != modified_no_diag.T).nnz),
        "degree_js_divergence": _js_divergence(original_hist, modified_hist),
        "mean_absolute_degree_shift": float(
            np.mean(np.abs(modified_degree - original_degree))
        ),
        "max_absolute_degree_shift": float(
            np.abs(modified_degree - original_degree).max()
            if len(original_degree)
            else 0.0
        ),
        "original_components": float(original_components),
        "modified_components": float(modified_components),
        "original_homophily": original_homophily,
        "modified_homophily": modified_homophily,
        "absolute_homophily_shift": abs(modified_homophily - original_homophily),
    }
    if include_transitivity:
        original_transitivity = _transitivity(original_no_diag)
        modified_transitivity = _transitivity(modified_no_diag)
        metrics.update(
            {
                "original_transitivity": original_transitivity,
                "modified_transitivity": modified_transitivity,
                "absolute_transitivity_shift": abs(
                    modified_transitivity - original_transitivity
                ),
            }
        )
    return metrics


def feature_metrics(
    original: sp.spmatrix, modified: sp.spmatrix, eps: float = 1e-9
) -> Dict[str, float]:
    original = original.tocsr().astype(np.float64)
    modified = modified.tocsr().astype(np.float64)
    original.eliminate_zeros()
    modified.eliminate_zeros()
    delta = (modified - original).tocsr()
    delta.data[np.abs(delta.data) <= eps] = 0.0
    delta.eliminate_zeros()
    changed_per_node = np.diff(delta.indptr)
    modified_nodes = np.flatnonzero(changed_per_node)

    dot = np.asarray(original.multiply(modified).sum(axis=1)).reshape(-1)
    original_norm = np.sqrt(
        np.asarray(original.multiply(original).sum(axis=1)).reshape(-1)
    )
    modified_norm = np.sqrt(
        np.asarray(modified.multiply(modified).sum(axis=1)).reshape(-1)
    )
    denominator = original_norm * modified_norm
    cosine = np.ones(original.shape[0], dtype=float)
    valid = denominator > eps
    cosine[valid] = dot[valid] / denominator[valid]
    only_one_zero = (original_norm <= eps) ^ (modified_norm <= eps)
    cosine[only_one_zero] = 0.0

    original_support = original.copy()
    modified_support = modified.copy()
    original_support.data[:] = 1.0
    modified_support.data[:] = 1.0
    intersection = original_support.multiply(modified_support)
    original_active = np.diff(original_support.indptr)
    modified_active = np.diff(modified_support.indptr)
    common_active = np.diff(intersection.tocsr().indptr)
    union = original_active + modified_active - common_active
    jaccard = np.ones(original.shape[0], dtype=float)
    union_mask = union > 0
    jaccard[union_mask] = common_active[union_mask] / union[union_mask]

    mean_original = np.asarray(original.mean(axis=0)).reshape(-1)
    mean_modified = np.asarray(modified.mean(axis=0)).reshape(-1)
    selected_cosine = cosine[modified_nodes] if len(modified_nodes) else np.asarray([1.0])
    selected_jaccard = jaccard[modified_nodes] if len(modified_nodes) else np.asarray([1.0])
    return {
        "modified_nodes": float(len(modified_nodes)),
        "modified_node_rate": float(len(modified_nodes) / max(original.shape[0], 1)),
        "changed_feature_entries": float(delta.nnz),
        "mean_changed_entries_per_modified_node": float(
            delta.nnz / max(len(modified_nodes), 1)
        ),
        "mean_cosine_similarity_modified_nodes": float(selected_cosine.mean()),
        "median_cosine_similarity_modified_nodes": float(
            np.median(selected_cosine)
        ),
        "min_cosine_similarity_modified_nodes": float(selected_cosine.min()),
        "mean_active_feature_jaccard_modified_nodes": float(selected_jaccard.mean()),
        "mean_added_features_per_modified_node": float(
            np.sum(modified_active - common_active) / max(len(modified_nodes), 1)
        ),
        "mean_removed_features_per_modified_node": float(
            np.sum(original_active - common_active) / max(len(modified_nodes), 1)
        ),
        "feature_mean_l2_shift": float(
            np.linalg.norm(mean_modified - mean_original)
        ),
    }


def analyze_artifact_dir(
    artifact_dir: Path,
    *,
    attack_mode: str,
    include_transitivity: bool = True,
) -> Dict[str, object]:
    original_adj = sp.load_npz(artifact_dir / "original_adj.npz")
    modified_adj = sp.load_npz(artifact_dir / "modified_adj.npz")
    original_features = sp.load_npz(artifact_dir / "original_features.npz")
    modified_features = sp.load_npz(artifact_dir / "modified_features.npz")
    labels = np.load(artifact_dir / "labels.npy")
    metadata_path = artifact_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    if attack_mode == "edge":
        modified_features = original_features
    elif attack_mode == "feature":
        modified_adj = original_adj
    elif attack_mode != "combined":
        raise ValueError(f"Unknown attack mode: {attack_mode}")
    return {
        "metadata": metadata,
        "attack_mode": attack_mode,
        "structural": structural_metrics(
            original_adj,
            modified_adj,
            labels,
            include_transitivity=include_transitivity,
        ),
        "feature": feature_metrics(original_features, modified_features),
        "semantic_scope": (
            "Feature-space metrics only. Text-level semantic similarity requires paired raw text."
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument(
        "--attack-mode", choices=["edge", "feature", "combined"], required=True
    )
    parser.add_argument("--skip-transitivity", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = analyze_artifact_dir(
        args.artifact_dir,
        attack_mode=args.attack_mode,
        include_transitivity=not args.skip_transitivity,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
