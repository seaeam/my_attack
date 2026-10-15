"""Offline checks for the versioned pre-flip, degree-source 2x2."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from heir import Heirattack
from gbhaa_experiments import run_followup as runner
from gbhaa_experiments import summarize_followup as summary
from gbhaa_experiments.entrypoints.meta_followup import FollowupAttack, validate_protocol_options
import test_followup_recording as recording


FACTORIAL_PROTOCOL = "aepr-factorial-v2-pre-stable"


class FactorialProtocolTests(unittest.TestCase):
    def test_factorial_jobs_use_new_pre_protocol_and_matched_settings(self):
        jobs = runner.make_jobs("factorial", range(15, 20), [.05, .1])
        self.assertEqual(len(jobs), 40)
        self.assertEqual({j["local_graph"] for j in jobs}, {"pre"})
        self.assertEqual({j["global_source"] for j in jobs}, {"degree"})
        self.assertEqual({(j["global_transition"], j["selector"]) for j in jobs},
                         {("topology", "ppr"), ("topology", "ae_pr"),
                          ("attribute", "ppr"), ("attribute", "ae_pr")})
        self.assertTrue(all(runner.settings(j) == runner.settings(jobs[0]) for j in jobs))
        self.assertEqual(runner.protocol_for_suite("factorial"), FACTORIAL_PROTOCOL)
        self.assertEqual(runner.protocol_for_suite("selectors"), "aepr-selectors-v2-pre-pr")
        self.assertEqual(runner.protocol_for_suite("ordering"), runner.PROTOCOL)

    def test_factorial_default_dry_run_is_isolated(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = runner.main(["--suite", "factorial", "--seeds", "15", "--budgets", ".05"])
        self.assertEqual(code, 0)
        commands = output.getvalue()
        self.assertIn("followup_results_factorial_pre", commands)
        self.assertIn("--followup-protocol " + FACTORIAL_PROTOCOL, commands)
        self.assertIn("--followup-local-graph pre", commands)
        self.assertNotIn("--followup-selector pr ", commands)

    def test_factorial_options_accept_only_endpoint_selectors_on_pre_graph(self):
        for selector in ("ppr", "ae_pr"):
            validate_protocol_options(SimpleNamespace(protocol=FACTORIAL_PROTOCOL,
                selector=selector, local_graph="pre", global_transition="topology"))
        for selector, graph in (("pr", "pre"), ("random", "pre"), ("ae_pr", "post")):
            with self.subTest(selector=selector, graph=graph), self.assertRaises(ValueError):
                validate_protocol_options(SimpleNamespace(protocol=FACTORIAL_PROTOCOL,
                    selector=selector, local_graph=graph, global_transition="attribute"))

    def test_legacy_and_selector_protocols_retain_exact_numpy_global_order(self):
        scores = np.tile([.1, .3, .2], 10)
        for identity in (runner.PROTOCOL, "aepr-selectors-v2-pre-pr"):
            for source in ("degree", "uniform"):
                attack = FollowupAttack.__new__(FollowupAttack)
                attack.options = SimpleNamespace(protocol=identity)
                attack.global_seed_strategy = source
                with self.subTest(protocol=identity, source=source):
                    np.testing.assert_array_equal(attack.rank_global_candidates(scores, 15),
                                                  np.argsort(-scores)[:15])

    def test_factorial_breaks_only_exact_ties_and_records_boundary_diagnostics(self):
        attack = FollowupAttack.__new__(FollowupAttack)
        attack.options = SimpleNamespace(protocol=FACTORIAL_PROTOCOL)
        attack.global_seed_strategy = "degree"
        attack.diagnostics = [{}]
        order = attack.rank_global_candidates(np.array([.1, .2, .2, .2 + 1e-9]), 2)
        np.testing.assert_array_equal(order, [3, 1])
        self.assertEqual(attack.diagnostics[0].get("global_unique_scores"), 3)
        self.assertEqual(attack.diagnostics[0]["global_cutoff_tie_count"], 2)
        with self.assertRaises(ValueError):
            attack.rank_global_candidates(np.array([.1, np.nan]), 1)

    def test_global_and_local_operators_preserve_their_distinct_readouts(self):
        attack = Heirattack.__new__(Heirattack)
        adj = sp.csr_matrix([[0., 1., 0., 0.], [1., 0., 1., 0.],
                             [0., 1., 0., 1.], [0., 0., 1., 0.]])
        features = torch.tensor([[1., 0.], [1., 0.], [.6, .8], [.6, .8]])
        for weighted, expected_global, expected_local in (
            (False, [9/44, 13/44, 13/44, 9/44], [.3, .5, .2, 0.]),
            (True, [6/29, 17/58, 17/58, 6/29], [.35, .5, .15, 0.]),
        ):
            with self.subTest(weighted=weighted):
                source = attack.compute_global_ppr(features, adj, alpha=.2, N=1,
                    seed_strategy="degree", use_cos_sim=weighted)
                np.testing.assert_allclose(source, [1/6, 1/3, 1/3, 1/6], atol=1e-7)
                scores = attack.compute_global_ppr(features, adj, alpha=.2, N=2,
                    seed_strategy="degree", use_cos_sim=weighted)
                np.testing.assert_allclose(scores, expected_global, atol=1e-7)
                _, local = attack.ppr_topk_from_seeds(features, adj, [0, 1], 4,
                    alpha=.2, N=1, use_cos_sim=weighted)
                np.testing.assert_allclose(local, expected_local, atol=1e-7)

    def test_factorial_rejects_changed_runtime_rules_before_writing_artifacts(self):
        for key, value in (("global_seed_strategy", "uniform"), ("freeze_structure_features", False),
                           ("global_ppr_alpha", .2), ("global_ppr_iters", 29),
                           ("text_ppr_alpha", .2), ("text_ppr_iters", 59)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                attack = recording.FollowupRecordingTests().make_attack(directory)
                attack.options = SimpleNamespace(protocol=FACTORIAL_PROTOCOL,
                    selector="ae_pr", local_graph="pre", global_transition="attribute")
                attack.global_ppr_iters = 30
                attack.text_ppr_alpha = .28
                attack.text_ppr_iters = 60
                setattr(attack, key, value)
                with self.assertRaisesRegex(ValueError, "Factorial protocol requires"):
                    attack.meta_attack_multi_step(np.ones((3, 2), dtype=np.float32),
                        sp.csr_matrix((3, 3)), np.array([0, 1, 0]),
                        np.array([0]), np.array([1, 2]), 1)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_real_loop_dispatches_global_ranking_and_records_factorial_artifacts(self):
        hashes = []
        for transition in ("topology", "attribute"):
            for selector in ("ppr", "ae_pr"):
                with self.subTest(global_kind=transition, selector=selector), \
                        tempfile.TemporaryDirectory() as directory:
                    artifacts = Path(directory) / "artifacts"
                    artifacts.mkdir()
                    attack = recording.FollowupRecordingTests().make_attack(artifacts)
                    attack.options = SimpleNamespace(protocol=FACTORIAL_PROTOCOL,
                        selector=selector, local_graph="pre", global_transition=transition)
                    attack.global_ppr_iters = 30
                    attack.text_ppr_alpha = .28
                    attack.text_ppr_iters = 60
                    attack.evaluations = {}
                    adj = sp.csr_matrix([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
                    labels, targets = np.array([0, 1, 0]), np.array([1, 2])
                    with patch.object(Heirattack, "inner_train", return_value=None), \
                            patch.object(attack, "rank_global_candidates",
                                         wraps=attack.rank_global_candidates) as rank:
                        attack.meta_attack_multi_step(np.ones((3, 2), dtype=np.float32),
                            adj, labels, np.array([0]), targets, 1)
                    rank.assert_called_once()
                    diagnostic = attack.diagnostics[0]
                    self.assertEqual(diagnostic["global_selected_nodes"], [1, 0, 2])
                    self.assertEqual(diagnostic["global_unique_scores"], 2)
                    self.assertEqual(diagnostic["global_cutoff_tie_count"], 2)
                    self.assertEqual(diagnostic["local_edge_value"], 0)
                    output = np.eye(2)[labels]
                    attack.record_victim_evaluation("Clean graph", output, labels, targets)
                    attack.record_victim_evaluation("Combined attack", output, labels, targets)
                    metadata, evaluation = runner.load_completed_artifacts(Path(directory), FACTORIAL_PROTOCOL)
                    self.assertEqual(metadata["local_graph"], "pre")
                    self.assertEqual(metadata["requested_local_graph"], "pre")
                    declaration = metadata.get("operator_settings")
                    self.assertIsNotNone(declaration)
                    self.assertEqual(declaration["global"]["source"], "unweighted_degree_on_current_graph")
                    self.assertEqual(declaration["global"]["accumulated_states"], 30)
                    self.assertEqual(declaration["local"]["updates"], 60)
                    self.assertEqual(declaration["global_ties"], "node_id_ascending")
                    hashes.append(metadata["actions_hash"])
                    for filename, record in (("attack.json", metadata), ("evaluation.json", evaluation)):
                        path = artifacts / filename
                        path.write_text(json.dumps({**record, "protocol": runner.PROTOCOL}))
                        with self.assertRaises(ValueError):
                            runner.load_completed_artifacts(Path(directory), FACTORIAL_PROTOCOL)
                        path.write_text(json.dumps(record))
                    path = artifacts / "attack.json"
                    altered = {**metadata, "operator_settings": {}}
                    path.write_text(json.dumps(altered))
                    with self.assertRaisesRegex(ValueError, "operator_settings"):
                        runner.load_completed_artifacts(Path(directory), FACTORIAL_PROTOCOL)
                    path.write_text(json.dumps(metadata))
                    (Path(directory) / "result.json").write_text(json.dumps({
                        "status": "complete", "attack": altered, "evaluation": evaluation}))
                    with self.assertRaisesRegex(ValueError, "differs"):
                        runner.load_completed_artifacts(Path(directory), FACTORIAL_PROTOCOL)
        self.assertEqual(len(set(hashes)), 1)  # Synthetic gradient fixes the action.

    def test_factorial_has_symmetric_global_and_local_effects_and_interaction(self):
        coefficients = summary.contrasts("factorial")
        self.assertIn("global_gain_with_topology_local", coefficients)
        self.assertIn("global_gain_with_attribute_local", coefficients)
        # Synthetic attacked accuracies, in percentage points.
        values = {"topology_ppr": 80, "topology_ae_pr": 70,
                  "attribute_ppr": 75, "attribute_ae_pr": 60}
        measured = {name: sum(weight * values[v] for v, weight in coef.items())
                    for name, coef in coefficients.items()}
        self.assertEqual(measured, {"global_gain_with_topology_local": 5,
            "global_gain_with_attribute_local": 10, "local_gain_with_topology_global": 10,
            "local_gain_with_attribute_global": 15, "interaction_in_accuracy_drop_pp": 5})


if __name__ == "__main__":
    unittest.main()
