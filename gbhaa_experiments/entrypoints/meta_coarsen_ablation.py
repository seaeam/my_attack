"""Run GB/K-Means/random hierarchy variants through meta_edge_only.py."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import heir as heir_module


REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_ATTACK = heir_module.Heirattack
_METHOD = "gb"


class RandomBalancedCluster:
    """Balanced random grouping with the same interface as GBCluster."""

    def __init__(self, n_clusters: int, seed: Optional[int] = None):
        self.n_clusters = max(1, int(n_clusters))
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.mode = "euclidean"
        self.centroids = None
        self.radii = None

    def fit_predict(self, embeddings, centroids=None):
        if torch.is_tensor(embeddings):
            values = embeddings.detach().cpu().numpy()
        else:
            values = np.asarray(embeddings)
        n = len(values)
        if n == 0:
            return torch.empty((0,), dtype=torch.long)
        k = min(self.n_clusters, n)
        permutation = self.rng.permutation(n)
        labels = np.empty(n, dtype=np.int64)
        labels[permutation] = np.arange(n, dtype=np.int64) % k
        centers = []
        radii = []
        for cluster_id in range(k):
            members = values[labels == cluster_id]
            center = members.mean(axis=0)
            centers.append(center)
            distances = np.linalg.norm(members - center, axis=1)
            radii.append(float(distances.max()) if len(distances) else 0.0)
        self.centroids = torch.from_numpy(np.asarray(centers)).float()
        self.radii = torch.tensor(radii, dtype=torch.float32)
        return torch.from_numpy(labels).long()


class CoarsenAblationAttack(_BASE_ATTACK):
    def __init__(self, *args, **kwargs):
        patched_class = heir_module.Heirattack
        heir_module.Heirattack = _BASE_ATTACK
        try:
            super().__init__(*args, **kwargs)
        finally:
            heir_module.Heirattack = patched_class

    def _make_clusterer(self, n_clusters, args=None):
        if _METHOD == "random":
            source = self.args if args is None else args
            return RandomBalancedCluster(n_clusters, seed=getattr(source, "seed", None))
        source = self.args if args is None else args
        source.coarsen_method = _METHOD
        return super()._make_clusterer(n_clusters, source)


def _consume_args() -> None:
    global _METHOD
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--experiment_coarsen_method",
        choices=["gb", "kmeans", "random"],
        required=True,
    )
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    _METHOD = parsed.experiment_coarsen_method
    # meta_edge_only.py only knows gb/kmeans. The subclass owns the actual method.
    sys.argv = [sys.argv[0], *remaining, "--coarsen_method", "gb"]


def main() -> None:
    _consume_args()
    print(f"Experiment coarsening method: {_METHOD}")
    heir_module.Heirattack = CoarsenAblationAttack
    runpy.run_path(str(REPO_ROOT / "meta_edge_only.py"), run_name="__main__")


if __name__ == "__main__":
    main()
