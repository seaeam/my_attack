"""Synthetic tests for the isolated Endpoint-Post development sweep."""
import json
import re
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from heir import Heirattack
from gbhaa_experiments import post_tuning_protocol


class PostTuningProtocolTests(unittest.TestCase):
    protocol = post_tuning_protocol

    def setUp(self):
        self.config = self.protocol.get_profile("baseline")
        self.adj = sp.csr_matrix([[0., 1., 1., 0.], [1., 0., 0., 1.],
                                  [1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.features = torch.tensor([[1., 1.], [1., .5], [1., 0.], [0., 1.]])

    def test_catalog_is_complete_json_and_returns_independent_configs(self):
        catalog = self.protocol.profile_catalog()
        self.assertEqual(self.protocol.PROTOCOL, "gbca-post-tuning-v1")
        self.assertEqual(len(catalog), 43)
        keys = {"id", "family", "description", "alpha", "iterations", "weight_floor",
                "visit_gamma", "readout", "generation_max_tokens", "text_min_cluster_size",
                "text_max_cluster_size", "text_max_added_words", "text_cdl_topk",
                "text_budget_per_node", "text_similarity_min"}
        for name, config in catalog.items():
            self.assertTrue(re.fullmatch(r"[a-z0-9_]+", name))
            self.assertEqual(config["id"], name)
            extra = {"selector_rule", "selector_pool_size"} if config["family"] == "selector_mechanism" else set()
            if config["family"] == "projected_editing":
                extra = {"editor_gain", "editor_gradient", "editor_candidates"}
            if config['family']=='editor_transfer':
                extra={'editor_gain','editor_gradient','editor_candidates','editor_proxy','editor_residual'}
            if config['family']=='editor_objective':
                extra={'editor_gain','editor_gradient','editor_candidates','editor_proxy','editor_residual','editor_objective','editor_local_mix'}
            self.assertEqual(set(config), keys | extra)
        self.assertEqual(json.loads(json.dumps(catalog)), catalog)
        self.assertEqual([self.config[key] for key in (
            "alpha", "iterations", "weight_floor", "visit_gamma", "readout",
            "generation_max_tokens", "text_min_cluster_size", "text_max_cluster_size",
            "text_max_added_words", "text_cdl_topk", "text_budget_per_node", "text_similarity_min")],
            [.28, 60, 0., 0., "last", 50, 2, 4, 70, 36, 80, .35])
        catalog["baseline"]["alpha"] = .99
        self.config["alpha"] = .88
        self.assertEqual(self.protocol.get_profile("baseline")["alpha"], .28)

    def test_quick_tiny_extended_and_explicit_trial_grids(self):
        for preset, expected in (("quick", 12), ("tiny", 6), ("extended", 22)):
            trials = self.protocol.build_trials(preset=preset)
            self.assertEqual(len(trials), expected)
            controls = [t for t in trials if t["arm"] != "endpoint_post"]
            self.assertEqual({t["arm"] for t in controls}, {"parallel", "endpoint_pre"})
            self.assertTrue(all(t["profile_id"] == "baseline" for t in controls))
            self.assertEqual(len({t["trial_id"] for t in trials}), expected)
            for trial in trials:
                self.assertEqual(trial["trial_id"], f'{trial["profile_id"]}__{trial["arm"]}')
                self.assertEqual(trial["config"], self.protocol.get_profile(trial["profile_id"]))
        self.assertEqual(self.protocol.select_profiles("tiny"),
                         ["baseline", "floor025", "coverage10", "combined"])
        selected = self.protocol.select_profiles(profile_ids=["combined"])
        self.assertEqual(selected, ["combined"])
        trials = self.protocol.build_trials(profile_ids=selected)
        self.assertEqual(len(trials), 4)
        self.assertEqual({(t["profile_id"], t["arm"]) for t in trials}, {
            ("combined", "endpoint_post"), ("baseline", "endpoint_post"),
            ("baseline", "endpoint_pre"), ("baseline", "parallel")})

    def test_matched_controls_apply_selected_configuration_to_all_arms(self):
        trials = self.protocol.build_trials(profile_ids=["combined"], matched_controls=True)
        self.assertEqual(len(trials), 4)
        self.assertEqual({(t["profile_id"], t["arm"]) for t in trials}, {
            ("combined", "endpoint_post"), ("combined", "endpoint_pre"),
            ("combined", "parallel"), ("baseline", "endpoint_post")})
        self.assertEqual(len(self.protocol.build_trials("tiny", matched_controls=True)), 12)
        trials[0]["config"]["alpha"] = .99
        self.assertTrue(all(t["config"]["alpha"] != .99 for t in trials[1:]))

    def test_job_grid_records_prefix_source_and_does_not_share_configs(self):
        jobs = self.protocol.make_jobs(["citeseer"], [15], .1, 60)
        self.assertEqual(len(jobs), 12)
        self.assertTrue(all(j["dataset"] == "citeseer" and j["seed"] == 15
                            and j["source_budget"] == .1 and j["max_actions"] == 60 for j in jobs))
        jobs = self.protocol.make_jobs(["citeseer", "cora"], [15, 16], .1, 1, preset="tiny")
        self.assertEqual(len(jobs), 24)
        jobs[0]["config"]["alpha"] = .99
        self.assertTrue(all(j["config"]["alpha"] != .99 for j in jobs[1:]))

    def test_profile_parameters_match_named_sweep(self):
        expected = {
            "alpha010": {"alpha": .1, "iterations": 120},
            "alpha015": {"alpha": .15, "iterations": 120},
            "alpha050": {"alpha": .5}, "floor010": {"weight_floor": .1},
            "floor025": {"weight_floor": .25}, "coverage05": {"visit_gamma": .5},
            "coverage10": {"visit_gamma": 1.}, "diffused": {"readout": "diffused"},
            "combined": {"alpha": .15, "iterations": 120, "weight_floor": .1,
                         "visit_gamma": .5, "readout": "diffused"},
            "alpha005": {"alpha": .05, "iterations": 240}, "alpha075": {"alpha": .75},
            "floor050": {"weight_floor": .5}, "coverage025": {"visit_gamma": .25},
            "combo_coverage": {"alpha": .15, "iterations": 120, "visit_gamma": 1.},
            "combo_floor": {"weight_floor": .1, "visit_gamma": 1.},
            "tokens150": {"generation_max_tokens": 150}, "groups2": {"text_max_cluster_size": 2},
            "added35": {"text_max_added_words": 35}, "cdl18": {"text_cdl_topk": 18},
        }
        for name, changes in expected.items():
            config = self.protocol.get_profile(name)
            for key, value in changes.items():
                self.assertEqual(config[key], value, (name, key))

    def test_runtime_settings_preserve_fixed_rules_and_apply_configurable_text_options(self):
        from gbhaa_experiments.action_state_protocol import protocol_settings
        expected = protocol_settings()
        config = self.protocol.get_profile("combined")
        for key in config:
            if key.startswith("text_"):
                expected[key] = config[key]
        expected.update(text_ppr_alpha=.15, text_ppr_iters=120)
        actual = self.protocol.runtime_settings(config)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["text_attack_nodes"], 4)
        self.assertEqual(actual["text_attack_max_visits"], 10)
        self.assertNotIn("generation_max_tokens", actual)
        actual["text_attack_nodes"] = 9
        self.assertEqual(self.protocol.runtime_settings(config)["text_attack_nodes"], 4)

    def test_operator_metadata_records_actual_diffusion_and_generation_settings(self):
        from gbhaa_experiments.action_state_protocol import operator_settings
        expected = operator_settings()
        expected.update(local_restart=.15, local_updates=120,
                        local_readout="max(last_state-alpha*q,0)",
                        attribute_weights="floor+(1-floor)*clipped_cosine", weight_floor=.1,
                        visit_gamma=.5, ranking="readout/(1+visits)**visit_gamma",
                        generation_max_tokens=50, scope="short_prefix_validation_development")
        self.assertEqual(self.protocol.operator_settings(self.protocol.get_profile("combined")), expected)
        self.assertEqual(self.protocol.operator_settings(self.config)["local_readout"], "last_state")
        self.assertEqual(self.protocol.operator_settings(self.protocol.get_profile("tokens150"))[
            "generation_max_tokens"], 150)

    def test_baseline_matches_existing_float32_diffusion_exactly(self):
        attack = Heirattack.__new__(Heirattack)
        for features in (self.features, self.features.to_sparse(), torch.zeros((4, 2))):
            for restarts in ([0, 1], [2], [0, 3]):
                _, expected = attack.ppr_topk_from_seeds(
                    features, self.adj, restarts, topk=4, alpha=.28, N=60, use_cos_sim=True)
                actual = self.protocol.diffusion_scores(features, self.adj, restarts, self.config)
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_array_equal(actual, expected)

    def test_floor_activates_zero_cosine_edges_and_preserves_inputs(self):
        features = torch.eye(2)
        adj = sp.csr_matrix([[0., 1.], [1., 0.]])
        baseline = self.protocol.diffusion_scores(features, adj, [0], self.config)
        floored = self.protocol.diffusion_scores(features, adj, [0], self.protocol.get_profile("floor025"))
        np.testing.assert_array_equal(baseline, [1., 0.])
        self.assertGreater(floored[1], .3)
        np.testing.assert_array_equal(adj.toarray(), [[0., 1.], [1., 0.]])
        np.testing.assert_array_equal(features, np.eye(2))

    def test_restart_probability_changes_scores(self):
        low = self.protocol.diffusion_scores(self.features, self.adj, [0], self.protocol.get_profile("alpha010"))
        high = self.protocol.diffusion_scores(self.features, self.adj, [0], self.protocol.get_profile("alpha050"))
        self.assertGreater(high[0], low[0])
        self.assertGreater(float(np.abs(high - low).sum()), .1)

    def test_visit_penalty_changes_top_selection_and_supports_mapping_and_dense(self):
        adj = sp.csr_matrix([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]])
        features = np.ones((3, 2))
        original, _ = self.protocol.score_nodes(features, adj, [0], self.config)
        config = self.protocol.get_profile("coverage10")
        order, scores = self.protocol.score_nodes(features, adj, [0], config, visits={0: 10})
        other_order, other_scores = self.protocol.score_nodes(features, adj, [0], config, visits=[10, 0, 0])
        self.assertEqual(original[0], 0)
        self.assertEqual(order[0], 1)
        np.testing.assert_array_equal(order, other_order)
        np.testing.assert_array_equal(scores, other_scores)

    def test_diffused_readout_removes_direct_endpoint_mass(self):
        adj = sp.csr_matrix([[0., 1.], [1., 0.]])
        features = np.ones((2, 2))
        last_order, last = self.protocol.score_nodes(features, adj, [0], self.config)
        order, scores = self.protocol.score_nodes(features, adj, [0], self.protocol.get_profile("diffused"))
        self.assertEqual(last_order[0], 0)
        self.assertEqual(order[0], 1)
        np.testing.assert_allclose(scores, [last[0] - .28, last[1]], rtol=0, atol=1e-7)

    def test_restarts_are_distinct_and_full_order_uses_stable_node_ties(self):
        one = self.protocol.diffusion_scores(self.features, self.adj, [0, 1], self.config)
        repeated = self.protocol.diffusion_scores(self.features, self.adj, [0, 0, 1], self.config)
        np.testing.assert_array_equal(one, repeated)
        order, scores = self.protocol.score_nodes(np.ones((4, 1)), sp.csr_matrix((4, 4)), [2], self.config)
        np.testing.assert_array_equal(order, [2, 0, 1, 3])
        self.assertEqual(scores.shape, (4,))

    def test_rejects_invalid_profile_choices_and_job_inputs(self):
        for kwargs in ({"preset": "missing"}, {"profile_ids": []}, {"profile_ids": ["bad"]},
                       {"profile_ids": ["baseline", "baseline"]}, {"profile_ids": "baseline"},
                       {"matched_controls": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.protocol.build_trials(**kwargs)
        with self.assertRaises(ValueError):
            self.protocol.get_profile("missing")
        for changes in ({"datasets": []}, {"datasets": ["cora", "cora"]}, {"datasets": ["bad"]},
                        {"seeds": []}, {"seeds": [15, 15]}, {"seeds": [-1]}, {"seeds": [True]},
                        {"seeds": [1.5]}, {"seeds": [2**32]}, {"source_budget": 0},
                        {"source_budget": 1.01}, {"source_budget": float("nan")},
                        {"source_budget": True}, {"max_actions": 0}, {"max_actions": 1.5},
                        {"max_actions": True}):
            args = {"datasets": ["citeseer"], "seeds": [15], "source_budget": .1, "max_actions": 60}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.protocol.make_jobs(**{**args, **changes})

    def test_rejects_invalid_configs_and_scorer_inputs(self):
        invalid = {"alpha": [0, 1.01, float("nan"), True], "iterations": [0, 2.5, True],
                   "weight_floor": [-.1, 1.1, float("inf")], "visit_gamma": [-1, float("nan")],
                   "readout": ["sum"], "generation_max_tokens": [0, False],
                   "text_min_cluster_size": [0, 5], "text_max_cluster_size": [1],
                   "text_max_added_words": [-1], "text_cdl_topk": [0], "text_budget_per_node": [-1],
                   "text_similarity_min": [-.1, 1.1], "id": ["../unsafe"],
                   "family": [""], "description": [""]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.protocol.diffusion_scores(self.features, self.adj, [0], {**self.config, key: value})
        with self.assertRaises(ValueError):
            self.protocol.diffusion_scores(self.features, self.adj, [0], {**self.config, "unknown": 1})
        for restarts in ([], [-1], [4], [1.5], [True]):
            with self.subTest(restarts=restarts), self.assertRaises(ValueError):
                self.protocol.diffusion_scores(self.features, self.adj, restarts, self.config)
        for features, adj in ((np.ones((3, 2)), self.adj), (np.full((4, 2), np.nan), self.adj),
                              (self.features, sp.csr_matrix((4, 3))),
                              (self.features, self.adj * -1), (self.features, self.adj * np.nan)):
            with self.subTest(features=features, adj=adj), self.assertRaises(ValueError):
                self.protocol.diffusion_scores(features, adj, [0], self.config)
        for visits in ({0: -1}, {4: 1}, {0: .5}, {True: 1}, [0, 1], [0, 1, 2, float("inf")]):
            with self.subTest(visits=visits), self.assertRaises(ValueError):
                self.protocol.score_nodes(self.features, self.adj, [0], self.config, visits=visits)


if __name__ == "__main__":
    unittest.main()
