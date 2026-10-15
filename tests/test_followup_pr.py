"""Ordinary PR is a uniform-source comparator, separate from endpoint PPR."""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from heir import Heirattack
from gbhaa_experiments import followup_protocol as protocol
from gbhaa_experiments import run_followup as runner
from gbhaa_experiments import summarize_followup as summary


class OrdinaryPageRankTests(unittest.TestCase):
    def setUp(self):
        self.attack = Heirattack.__new__(Heirattack)
        self.star = sp.csr_matrix([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]])

    def select(self, adj, endpoints=(0, 1), features=None, strategy="pr"):
        if features is None:
            features = torch.ones((adj.shape[0], 2))
        return protocol.selector_order(self.attack, features, adj, endpoints,
                                       strategy, .2, 1, 15)

    def test_pr_matches_analytic_uniform_source_stationary_solution(self):
        original = self.star.copy()
        order, scores = self.select(self.star)
        np.testing.assert_allclose(scores, [13/27, 7/27, 7/27], atol=1e-10)
        np.testing.assert_array_equal(order, [0, 1, 2])
        self.assertAlmostEqual(float(scores.sum()), 1.)
        self.assertEqual((self.star != original).nnz, 0)

    def test_pr_ignores_endpoints_and_features_but_ppr_keeps_endpoint_restart(self):
        _, first = self.select(self.star)
        _, second = self.select(self.star, endpoints=(1, 2), features=torch.eye(3))
        np.testing.assert_array_equal(first, second)
        _, ppr = self.select(self.star, strategy="ppr")
        np.testing.assert_allclose(ppr, [.5, .3, .2], atol=1e-6)
        self.assertFalse(np.allclose(first, ppr))

    def test_pr_redistributes_isolated_mass_uniformly(self):
        adj = sp.csr_matrix([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
        order, scores = self.select(adj)
        np.testing.assert_allclose(scores, [5/11, 5/11, 1/11], atol=1e-10)
        np.testing.assert_array_equal(order, [0, 1, 2])
        order, scores = self.select(sp.csr_matrix((4, 4)))
        np.testing.assert_allclose(scores, np.full(4, .25))
        np.testing.assert_array_equal(order, np.arange(4))

    def test_pr_fails_if_solver_does_not_converge(self):
        with patch.object(protocol, "PR_MAX_ITER", 1, create=True), \
                self.assertRaises(nx.PowerIterationFailedConvergence):
            self.select(self.star)


class PRComparisonPlanTests(unittest.TestCase):
    def test_entrypoint_rejects_pr_with_legacy_protocol_or_post_graph(self):
        from gbhaa_experiments.entrypoints.meta_followup import validate_protocol_options
        for identity, selector, graph in ((runner.PROTOCOL, "pr", "pre"),
                                          ("aepr-selectors-v2-pre-pr", "pr", "post"),
                                          ("aepr-selectors-v2-pre-pr", "ppr", "pre")):
            with self.subTest(protocol=identity, selector=selector, graph=graph), self.assertRaises(ValueError):
                validate_protocol_options(SimpleNamespace(protocol=identity, selector=selector, local_graph=graph))
        validate_protocol_options(SimpleNamespace(protocol="aepr-selectors-v2-pre-pr", selector="pr", local_graph="pre"))

    def test_selector_matrix_uses_pr_and_pre_with_matched_settings(self):
        jobs = runner.make_jobs("selectors", [15, 16, 17, 18, 19], [.05, .1])
        self.assertEqual(len(jobs), 40)
        self.assertEqual({j["selector"] for j in jobs}, {"ae_pr", "pr", "degree", "random"})
        self.assertEqual({j["local_graph"] for j in jobs}, {"pre"})
        self.assertTrue(all(runner.settings(j) == runner.settings(jobs[0]) for j in jobs))

    def test_factorial_keeps_endpoint_ppr_and_separate_protocol(self):
        jobs = runner.make_jobs("factorial", [15], [.05])
        self.assertEqual({j["selector"] for j in jobs}, {"ppr", "ae_pr"})
        self.assertNotEqual(runner.protocol_for_suite("factorial"), runner.protocol_for_suite("selectors"))
        self.assertNotEqual(runner.protocol_for_suite("selectors"), runner.PROTOCOL)

    def test_selector_dry_run_uses_new_output_and_protocol(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = runner.main(["--suite", "selectors", "--seeds", "15", "--budgets", ".05"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("followup_results_pr_pre", text)
        self.assertIn("--followup-local-graph pre", text)
        self.assertIn("--followup-selector pr ", text)
        self.assertNotIn("--followup-selector ppr ", text)
        self.assertIn("--followup-protocol aepr-selectors-v2-pre-pr", text)

    def test_selector_contrast_is_pr_while_factorial_keeps_ppr(self):
        self.assertIn("pr_minus_ae_pr", summary.contrasts("selectors"))
        self.assertNotIn("ppr_minus_ae_pr", summary.contrasts("selectors"))
        self.assertIn("topology_ppr", summary.contrasts("factorial")["local_gain_with_topology_global"])


if __name__ == "__main__":
    unittest.main()
