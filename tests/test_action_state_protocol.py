import tempfile
import json
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from heir import Heirattack
from gbhaa_experiments.action_state_protocol import (
    ARMS, independent_restarts, flip_action, rank_on_graph, protocol_settings,
    make_jobs, validate_trajectory, actions_hash,
)
from gbhaa_experiments.action_state_cache import RequestCache


class ActionStateProtocolTests(unittest.TestCase):
    def setUp(self):
        self.adj = sp.csr_matrix([[0., 1., 0., 0.], [1., 0., 1., 0.],
                                  [0., 1., 0., 1.], [0., 0., 1., 0.]])
        self.features = torch.ones((4, 2))
        self.attack = Heirattack.__new__(Heirattack)

    def test_plan_has_one_structure_trajectory_for_three_arms(self):
        jobs = make_jobs(["citeseer", "cora"], range(15, 20), [.05, .1])
        self.assertEqual(len(jobs), 80)
        self.assertEqual(sum(j["variant"] == "structure" for j in jobs), 20)
        self.assertEqual([j["variant"] for j in jobs[:4]], ["structure", *ARMS])
        cfg = protocol_settings()
        self.assertEqual(cfg["global_seed_strategy"], "degree")
        self.assertEqual(cfg["text_attack_nodes"], 4)
        self.assertEqual(cfg["text_attack_max_visits"], 10)

    def test_parallel_restarts_use_only_seed_round_and_node_universe(self):
        pair = independent_restarts(100, 15, 0)
        self.assertEqual(len(set(pair)), 2)
        self.assertEqual(pair, independent_restarts(100, 15, 0))
        self.assertNotEqual(pair, independent_restarts(100, 15, 1))

    def test_flip_addition_and_deletion_are_exact_and_preserve_snapshot(self):
        post = flip_action(self.adj, [0, 3, 1])
        self.assertEqual(self.adj[0, 3], 0)
        self.assertEqual(post[0, 3], 1)
        self.assertEqual(post[3, 0], 1)
        restored = flip_action(post, [0, 3, 0])
        self.assertEqual((restored != self.adj).nnz, 0)
        with self.assertRaises(ValueError):
            flip_action(self.adj, [0, 3, 0])

    def test_same_endpoints_can_have_different_pre_post_scores(self):
        post = flip_action(self.adj, [1, 2, 0])
        pre_order, pre_scores = rank_on_graph(self.attack, self.features, self.adj, [1, 2])
        post_order, post_scores = rank_on_graph(self.attack, self.features, post, [1, 2])
        self.assertGreater(float(np.abs(pre_scores - post_scores).sum()), .01)
        self.assertEqual(sorted(pre_order.tolist()), list(range(4)))
        self.assertEqual(sorted(post_order.tolist()), list(range(4)))

    def test_trajectory_rejects_reflips_and_wrong_final_graph(self):
        acts = [[0, 3, 1], [1, 2, 0]]
        final = flip_action(flip_action(self.adj, acts[0]), acts[1])
        validate_trajectory(self.adj, final, acts, 2)
        self.assertEqual(actions_hash(acts), actions_hash(acts))
        with self.assertRaises(ValueError):
            validate_trajectory(self.adj, final, acts + [[0, 3, 0]], 3)
        with self.assertRaises(ValueError):
            validate_trajectory(self.adj, self.adj, acts, 2)

    def test_cache_keys_include_full_request_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = RequestCache(Path(directory))
            calls = []
            def generate():
                calls.append(1)
                return "one ||| two"
            request = {"messages": [{"role": "user", "content": "prompt"}],
                       "model": "offline", "seed": 15, "temperature": 0.}
            self.assertEqual(cache.get_or_generate(request, generate), "one ||| two")
            self.assertEqual(cache.get_or_generate(request, generate), "one ||| two")
            cache.get_or_generate({**request, "seed": 16}, generate)
            self.assertEqual(len(calls), 2)
            self.assertEqual(cache.stats["hits"], 1)
            saved = next(p for p in Path(directory).glob('*.json')
                         if json.loads(p.read_text())['request'] == request)
            saved.write_text('{"response": "corrupt"}')
            with self.assertRaises(ValueError):
                cache.get_or_generate(request, generate)

    def test_unparseable_reply_does_not_poison_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = RequestCache(directory)
            request = {'messages': [{'role': 'user', 'content': 'templates'}]}
            with self.assertRaises(ValueError):
                cache.get_or_generate(request, lambda: ' ||| ||| ')
            self.assertEqual(list(Path(directory).glob('*.json')), [])
            retried = RequestCache(directory)
            self.assertEqual(retried.get_or_generate(request, lambda: 'valid template'),
                             'valid template')


if __name__ == "__main__":
    unittest.main()
