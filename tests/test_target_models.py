import unittest
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from target_models import (
    GINNodeClassifier,
    GraphSAGENodeClassifier,
    build_pyg_data,
    build_target_model,
    train_and_evaluate_target,
)


class TargetModelTest(unittest.TestCase):
    def setUp(self):
        self.features = sp.csr_matrix(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.adj = sp.csr_matrix(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.labels = torch.tensor([0, 1, 0, 1])

    def test_build_pyg_data_preserves_features_edges_and_masks(self):
        data = build_pyg_data(
            adj=self.adj,
            features=self.features,
            labels=self.labels,
            idx_train=np.array([0, 1]),
            idx_val=np.array([2]),
        )

        self.assertEqual(tuple(data.x.shape), (4, 3))
        self.assertTrue(torch.equal(data.x, torch.tensor(self.features.toarray())))
        self.assertEqual(data.edge_index.shape[1], self.adj.nnz)
        self.assertEqual(data.train_mask.tolist(), [True, True, False, False])
        self.assertEqual(data.val_mask.tolist(), [False, False, True, False])

    def test_build_pyg_data_accepts_torch_sparse_csr_inputs(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            features = torch.tensor(self.features.toarray()).to_sparse_csr()
            adj = torch.tensor(self.adj.toarray()).to_sparse_csr()

        data = build_pyg_data(
            adj=adj,
            features=features,
            labels=self.labels,
            idx_train=np.array([0, 1]),
            idx_val=np.array([2]),
        )

        self.assertEqual(data.x.layout, torch.strided)
        self.assertEqual(data.edge_index.shape[1], self.adj.nnz)

    def test_build_pyg_data_ignores_explicit_zero_sparse_edges(self):
        adj = sp.coo_matrix(
            (
                np.array([1.0, 0.0, 1.0], dtype=np.float32),
                (np.array([0, 1, 2]), np.array([1, 2, 3])),
            ),
            shape=(4, 4),
        )

        data = build_pyg_data(
            adj=adj,
            features=self.features,
            labels=self.labels,
            idx_train=np.array([0, 1]),
            idx_val=np.array([2]),
        )

        self.assertEqual(data.edge_index.t().tolist(), [[0, 1], [2, 3]])

    def test_build_target_model_selects_gin_and_graphsage(self):
        gin = build_target_model("gin", nfeat=3, nhid=4, nclass=2, dropout=0.0)
        gsage = build_target_model(
            "gsage", nfeat=3, nhid=4, nclass=2, dropout=0.0
        )

        self.assertIsInstance(gin, GINNodeClassifier)
        self.assertIsInstance(gsage, GraphSAGENodeClassifier)

    def test_target_models_return_node_log_probabilities(self):
        data = build_pyg_data(
            adj=self.adj,
            features=self.features,
            labels=self.labels,
            idx_train=np.array([0, 1]),
            idx_val=np.array([2]),
        )

        for model_name in ("gin", "gsage"):
            model = build_target_model(
                model_name, nfeat=3, nhid=4, nclass=2, dropout=0.0
            )
            model.eval()
            output = model(data.x, data.edge_index)

            self.assertEqual(tuple(output.shape), (4, 2))
            self.assertTrue(
                torch.allclose(
                    torch.logsumexp(output, dim=1),
                    torch.zeros(4),
                    atol=1e-6,
                )
            )

    def test_target_models_can_train_and_evaluate_on_a_small_graph(self):
        for model_name in ("gin", "gsage"):
            result = train_and_evaluate_target(
                target_model_name=model_name,
                adj=self.adj,
                features=self.features,
                labels=self.labels,
                idx_train=np.array([0, 1]),
                idx_val=np.array([2]),
                idx_eval=np.array([2, 3]),
                device=torch.device("cpu"),
                hidden=4,
                dropout=0.0,
                epochs=3,
                patience=2,
                seed=15,
            )

            self.assertGreaterEqual(result.accuracy, 0.0)
            self.assertLessEqual(result.accuracy, 1.0)
            self.assertTrue(np.isfinite(result.loss))


if __name__ == "__main__":
    unittest.main()
