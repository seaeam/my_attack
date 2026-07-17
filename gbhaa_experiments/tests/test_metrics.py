import unittest

import numpy as np
import scipy.sparse as sp

from gbhaa_experiments.gradient_metrics import summarize_rows
from gbhaa_experiments.stealthiness import feature_metrics, structural_metrics


class MetricTest(unittest.TestCase):
    def test_coarse_gradient_summary_detects_perfect_ranking(self):
        rows = [
            {
                "coarse_score": 3.0,
                "fine_max_score": 30.0,
                "action_agreement": True,
            },
            {
                "coarse_score": 2.0,
                "fine_max_score": 20.0,
                "action_agreement": True,
            },
            {
                "coarse_score": 1.0,
                "fine_max_score": 10.0,
                "action_agreement": False,
            },
        ]
        summary = summarize_rows(rows, topk=2)
        self.assertAlmostEqual(summary["spearman"], 1.0)
        self.assertAlmostEqual(summary["kendall"], 1.0)
        self.assertEqual(summary["top1_hit"], 1.0)
        self.assertEqual(summary["topk_overlap"], 1.0)
        self.assertAlmostEqual(summary["action_agreement"], 2.0 / 3.0)

    def test_structural_metrics_count_one_undirected_flip(self):
        original = sp.csr_matrix(
            np.asarray(
                [
                    [0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                ]
            )
        )
        modified = original.copy().tolil()
        modified[0, 3] = 1
        modified[3, 0] = 1
        metrics = structural_metrics(
            original,
            modified.tocsr(),
            np.asarray([0, 0, 1, 1]),
            include_transitivity=False,
        )
        self.assertEqual(metrics["edge_flips"], 1.0)
        self.assertEqual(metrics["asymmetry_entries"], 0.0)
        self.assertAlmostEqual(metrics["edge_flip_rate"], 1.0 / 3.0)

    def test_explicit_sparse_zero_is_not_revived_as_an_edge(self):
        original = sp.csr_matrix(np.asarray([[0, 1], [1, 0]], dtype=float))
        modified = original.copy()
        modified[0, 1] = 0
        modified[1, 0] = 0
        metrics = structural_metrics(original, modified, np.asarray([0, 0]))
        self.assertEqual(metrics["edge_flips"], 1.0)
        self.assertEqual(metrics["modified_edges"], 0.0)
        self.assertEqual(metrics["asymmetry_entries"], 0.0)
        self.assertAlmostEqual(metrics["edge_flip_rate"], 1.0)

    def test_feature_metrics_report_modified_node_and_cosine(self):
        original = sp.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 1.0]]))
        modified = sp.csr_matrix(np.asarray([[1.0, 1.0], [0.0, 1.0]]))
        metrics = feature_metrics(original, modified)
        self.assertEqual(metrics["modified_nodes"], 1.0)
        self.assertEqual(metrics["changed_feature_entries"], 1.0)
        self.assertAlmostEqual(
            metrics["mean_cosine_similarity_modified_nodes"], 1.0 / np.sqrt(2.0)
        )


if __name__ == "__main__":
    unittest.main()
