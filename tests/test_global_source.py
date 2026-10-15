"""Offline checks for the main global AE-PR source configuration."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph.defense import GCN

from heir import Heirattack


ROOT = Path(__file__).resolve().parents[1]


class GlobalSourceTests(unittest.TestCase):
    def setUp(self):
        self.adj = sp.csr_matrix(
            [[0., 1., 0., 0.], [1., 0., 1., 0.],
             [0., 1., 0., 1.], [0., 0., 1., 0.]]
        )
        self.features = torch.tensor([[1., 0.], [1., 0.], [.6, .8], [.6, .8]])
        self.attack = Heirattack.__new__(Heirattack)
        self.degree_source = [1/6, 1/3, 1/3, 1/6]

    def test_default_source_uses_unweighted_current_graph_degrees(self):
        scores = self.attack.compute_global_ppr(self.features, self.adj, N=1)
        np.testing.assert_allclose(scores, self.degree_source, atol=1e-7)

    def test_constructor_without_source_option_uses_degree(self):
        surrogate = GCN(nfeat=2, nhid=2, nclass=2, device="cpu")
        attack = Heirattack(
            model=surrogate, nnodes=4, args=SimpleNamespace(seed=15),
            gb_data=None, features=self.features, feature_shape=(4, 2),
        )
        scores = attack.compute_global_ppr(
            self.features, self.adj, N=1, seed_strategy=attack.global_seed_strategy
        )
        np.testing.assert_allclose(scores, self.degree_source, atol=1e-7)

    def test_entrypoint_defaults_produce_degree_source(self):
        for filename in ("meta.py", "meta_gin.py", "meta_gsage.py", "meta_edge_only.py"):
            with self.subTest(entrypoint=filename):
                # Execute the real option declaration without the entrypoints'
                # top-level data loading, model training or generation calls.
                tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
                declaration = next(
                    node for node in tree.body
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "add_argument"
                    and node.value.args
                    and isinstance(node.value.args[0], ast.Constant)
                    and node.value.args[0].value == "--global_seed_strategy"
                )
                parser = argparse.ArgumentParser()
                exec(compile(ast.Module(body=[declaration], type_ignores=[]), filename, "exec"),
                     {"parser": parser})
                args = parser.parse_args([])
                scores = self.attack.compute_global_ppr(
                    self.features, self.adj, N=1, seed_strategy=args.global_seed_strategy
                )
                np.testing.assert_allclose(scores, self.degree_source, atol=1e-7)
                self.assertEqual(
                    parser.parse_args(["--global_seed_strategy", "uniform"]).global_seed_strategy,
                    "uniform",
                )

    def test_degree_source_falls_back_to_uniform_without_edges(self):
        scores = self.attack.compute_global_ppr(self.features, sp.csr_matrix((4, 4)))
        np.testing.assert_allclose(scores, np.full(4, .25), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
