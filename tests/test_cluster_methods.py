import unittest
from types import SimpleNamespace

import torch

from gb_division_simple import GBCluster, KMeansCluster
from heir import Heirattack


class ClusterMethodTest(unittest.TestCase):
    def test_kmeans_cluster_matches_gb_cluster_interface(self):
        embeddings = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 10.0],
                [10.1, 10.0],
            ],
            dtype=torch.float32,
        )
        clusterer = KMeansCluster(n_clusters=2, random_state=15)

        cluster_ids = clusterer.fit_predict(embeddings)

        self.assertEqual(tuple(cluster_ids.shape), (4,))
        self.assertEqual(cluster_ids.dtype, torch.long)
        self.assertEqual(tuple(clusterer.centroids.shape), (2, 2))

    def test_make_clusterer_selects_requested_method(self):
        args = SimpleNamespace(coarsen_method="kmeans", seed=15)
        clusterer = Heirattack._make_clusterer(
            object.__new__(Heirattack),
            n_clusters=2,
            args=args,
        )

        self.assertIsInstance(clusterer, KMeansCluster)

        args.coarsen_method = "gb"
        clusterer = Heirattack._make_clusterer(
            object.__new__(Heirattack),
            n_clusters=2,
            args=args,
        )

        self.assertIsInstance(clusterer, GBCluster)


if __name__ == "__main__":
    unittest.main()
