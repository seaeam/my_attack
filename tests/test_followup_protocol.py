import unittest

import numpy as np
import scipy.sparse as sp
import torch
from types import SimpleNamespace
from unittest.mock import patch

from heir import Heirattack


class FollowupProtocolTests(unittest.TestCase):
    def test_counted_retry_loop_owns_all_transport_requests(self):
        from gbhaa_experiments.followup_protocol import configure_generation
        from text_attack_generator import TextAttackGenerator
        import httpx
        from openai import OpenAI
        requests=[]
        def handle(request):
            requests.append(request)
            if len(requests)<3:
                return httpx.Response(500,json={"error":{"message":"temporary","type":"server_error"}})
            return httpx.Response(200,json={"id":"offline","created":0,"object":"chat.completion",
                "model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})
        generator=TextAttackGenerator.__new__(TextAttackGenerator)
        client=OpenAI(api_key="test",base_url="http://no-network.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        generator.client=client;generator.num_retries=2;generator.model_name="test";generator.max_tokens=8
        configure_generation(generator,15)
        with patch("text_attack_generator.time.sleep"):
            generator._generate_text_with_api_retry([])
        self.assertEqual(len(requests),3)
        self.assertEqual(generator.api_stats,{"attempts":3,"successes":1,"failures":2,"retries":2})
        client.close()

    def test_parsed_oracle_is_rejected_before_base_initialization(self):
        from gbhaa_experiments.entrypoints.meta_followup import FollowupAttack
        with self.assertRaisesRegex(ValueError, "Oracle"):
            FollowupAttack(args=SimpleNamespace(oracle=True))

    def test_edge_deletion_precedes_local_selection_as_well(self):
        attack = Heirattack.__new__(Heirattack)
        adj = sp.csr_matrix(np.array([[0,1], [1,0]]))
        updated = attack.prepare_attribute_adjacency(adj, 0, 1)
        self.assertEqual(updated.nnz, 0)
        self.assertEqual(adj.nnz, 2)

    def test_generation_seed_and_actual_api_attempts_are_recorded(self):
        from text_attack_generator import TextAttackGenerator
        generator = TextAttackGenerator.__new__(TextAttackGenerator)
        seen = []
        def create(**kwargs):
            seen.append(kwargs)
            return "response"
        generator.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        generator.num_retries=0; generator.max_tokens=8; generator.model_name="test"
        generator.generation_seed=15; generator.generation_temperature=0.
        self.assertEqual(generator._generate_text_with_api_retry([]),"response")
        self.assertEqual(seen[0]["seed"],15)
        self.assertEqual(generator.api_stats["attempts"],1)
        self.assertEqual(generator.api_stats["successes"],1)

    def test_real_loop_selects_and_embeds_on_flipped_graph(self):
        # Exercise the real state-transition loop with fixed synthetic gradients;
        # no training, dataset download or generator call occurs.
        attack = Heirattack.__new__(Heirattack)
        torch.nn.Module.__init__(attack)
        attack.nnodes = 3; attack.nclass = 2; attack.nfeat = 2; attack.M = 3
        attack.device = "cpu"; attack.freeze_structure_features = True
        attack.global_important_ratio = .1; attack.global_ppr_alpha = .15
        attack.global_ppr_iters = 2; attack.global_seed_strategy = "degree"
        attack.attack_structure = True; attack.attack_features = True
        attack.use_text_attack = True; attack.text_generator = object()
        attack.text_attack_max_visits = 1; attack.text_attack_total_nodes = None
        attack.text_min_cluster_size = 1; attack.text_max_cluster_size = 2
        attack.text_budget_per_node = 1
        attack.self_training_label = lambda labels, idx: labels
        attack.inner_train = lambda *a: None
        seen_norms = []
        def embeddings(features, norm):
            seen_norms.append(norm.to_dense().clone())
            return features
        attack.get_embeddings = embeddings
        gradient = torch.zeros((3, 3)); gradient[0, 2] = 5
        attack.get_meta_grad = lambda *a: (gradient, None)
        seen_graphs = []
        def select(fea, adj, row_idx, col_idx):
            seen_graphs.append(adj.copy())
            self.assertEqual(attack.full_adj.to_dense()[0, 2].item(), 1)
            return np.array([0])
        attack.select_text_candidate_nodes = select
        attack._build_round_text_clusters = lambda *a, **k: {0: [0]}
        attack.attack_features_with_text = lambda **k: k["full_features"]
        adj = sp.csr_matrix(np.array([[0,1,0], [1,0,1], [0,1,0]], dtype=np.float32))
        attack.meta_attack_multi_step(np.ones((3, 2), dtype=np.float32), adj,
            np.array([0,1,0]), np.array([0]), np.array([1,2]), 1)
        self.assertEqual(seen_graphs[0][0, 2], 1)
        self.assertFalse(torch.allclose(seen_norms[0], seen_norms[1]))
        self.assertEqual(attack.modified_adj[0, 2], 1)  # no second flip
        self.assertEqual((attack.modified_adj != adj).nnz, 2)

    def test_default_state_flips_before_local_selection_without_mutating_input(self):
        attack = Heirattack.__new__(Heirattack)
        adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        updated = attack.prepare_attribute_adjacency(adj, 0, 2)
        self.assertEqual(updated[0, 2], 1)
        self.assertEqual(updated[2, 0], 1)
        self.assertEqual(adj[0, 2], 0)

    def test_explicit_pre_flip_diagnostic_uses_old_graph(self):
        attack = Heirattack.__new__(Heirattack)
        attack.followup_edge_first = False
        adj = sp.csr_matrix((3, 3))
        self.assertIs(attack.prepare_attribute_adjacency(adj, 0, 1), adj)

    def test_local_ppr_matches_transposed_column_propagation(self):
        from gbhaa_experiments.followup_protocol import selector_order
        attack = Heirattack.__new__(Heirattack)
        adj = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float))
        order, scores = selector_order(attack, torch.eye(3), adj, [0, 1],
                                       "ppr", .2, 1, 15)
        np.testing.assert_allclose(scores, [.5, .3, .2], atol=1e-6)
        np.testing.assert_array_equal(order, [0, 1, 2])

    def test_all_selectors_share_full_universe_and_ties_use_node_id(self):
        from gbhaa_experiments.followup_protocol import selector_order
        attack = Heirattack.__new__(Heirattack)
        adj = sp.csr_matrix((6, 6))
        for strategy in ("ae_pr", "ppr", "degree", "random"):
            order, _ = selector_order(attack, torch.eye(6), adj, [2, 3], strategy, .2, 2, 4)
            self.assertEqual(sorted(order.tolist()), list(range(6)))
        order, _ = selector_order(attack, torch.eye(6), adj, [2, 3], "degree", .2, 2, 4)
        np.testing.assert_array_equal(order, np.arange(6))

    def test_uniform_global_is_constant_but_degree_is_not(self):
        attack = Heirattack.__new__(Heirattack)
        adj = sp.csr_matrix(np.array([[0,1,1,1], [1,0,0,0], [1,0,0,0], [1,0,0,0]], dtype=float))
        feature = torch.ones((4, 2))
        uniform = attack.compute_global_ppr(feature, adj, seed_strategy="uniform")
        degree = attack.compute_global_ppr(feature, adj, seed_strategy="degree")
        self.assertLess(float(np.ptp(uniform)), 1e-6)
        self.assertGreater(float(np.ptp(degree)), .01)

    def test_legacy_uniform_control_keeps_actual_argsort_behavior(self):
        from gbhaa_experiments.entrypoints.meta_followup import FollowupAttack
        from gbhaa_experiments.followup_config import PROTOCOL
        attack = FollowupAttack.__new__(FollowupAttack)
        attack.options = SimpleNamespace(protocol=PROTOCOL)
        attack.global_seed_strategy = "uniform"
        scores = np.array([.25-1e-8, .25+1e-8, .25, .25])
        np.testing.assert_array_equal(attack.rank_global_candidates(scores, 4), np.argsort(-scores))

    def test_metrics_distinguish_original_errors_from_attack_flips(self):
        from gbhaa_experiments.followup_protocol import prediction_metrics
        clean = np.array([0, 0, 1, 1]); adv = np.array([1, 1, 1, 0]); y = np.array([0, 1, 1, 0])
        out = prediction_metrics(clean, adv, y)
        self.assertEqual(out["clean_accuracy"], .5)
        self.assertEqual(out["attacked_accuracy"], .75)
        self.assertEqual(out["conditional_asr"], .5)
        self.assertEqual(out["clean_correct_count"], 2)

    def test_no_clean_correct_has_undefined_asr(self):
        from gbhaa_experiments.followup_protocol import prediction_metrics
        self.assertIsNone(prediction_metrics(np.array([0]), np.array([0]), np.array([1]))["conditional_asr"])


if __name__ == "__main__":
    unittest.main()
