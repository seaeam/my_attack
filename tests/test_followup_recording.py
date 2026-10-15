"""Offline regression for action/timing records in the existing pre-flip loop."""

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
from gbhaa_experiments.entrypoints.meta_followup import FollowupAttack
from gbhaa_experiments.followup_config import PROTOCOL, SELECTOR_PROTOCOL, pr_solver_settings
from gbhaa_experiments.run_followup import load_completed_artifacts


class FollowupRecordingTests(unittest.TestCase):
    def make_attack(self, directory, *, features_enabled=True, exhausted=False):
        attack = FollowupAttack.__new__(FollowupAttack)
        torch.nn.Module.__init__(attack)
        attack.args = SimpleNamespace(seed=15, dataset="synthetic")
        attack.options = SimpleNamespace(local_graph="post", selector="degree",
                                         global_transition="attribute")
        attack.followup_edge_first = True  # Must not change the existing loop order.
        attack.trace_dir = Path(directory)
        attack.round_index = -1
        attack.actions = []
        attack.diagnostics = []
        attack.attribute_counts = {}
        attack.nnodes = 3
        attack.nclass = 2
        attack.nfeat = 2
        attack.M = 3
        attack.device = "cpu"
        attack.freeze_structure_features = True
        attack.global_important_ratio = .1
        attack.global_ppr_alpha = .15
        attack.global_ppr_iters = 2
        attack.global_seed_strategy = "degree"
        attack.attack_structure = True
        attack.attack_features = features_enabled
        attack.use_text_attack = True
        attack.text_generator = SimpleNamespace(model_name="offline", api_stats={})
        attack.text_attack_max_visits = 1
        attack.text_attack_total_nodes = 0 if exhausted else None
        attack.text_attack_nodes = 1
        attack.text_topk_ratio = .1
        attack.text_ppr_alpha = .2
        attack.text_ppr_iters = 2
        attack.text_min_cluster_size = 1
        attack.text_max_cluster_size = 2
        attack.text_budget_per_node = 1
        attack.self_training_label = lambda labels, idx: labels
        attack.get_embeddings = lambda features, norm: features
        gradient = torch.zeros((3, 3))
        gradient[0, 2] = 5
        attack.get_meta_grad = lambda *args: (gradient, None)
        attack._build_round_text_clusters = lambda nodes, *a, **k: {0: nodes.tolist()}
        attack.attack_features_with_text = lambda **kwargs: kwargs["full_features"]
        return attack

    def test_all_pr_comparison_arms_record_new_protocol_on_same_pre_flip_graph(self):
        hashes = []
        for selector in ("ae_pr", "pr", "degree", "random"):
            with self.subTest(selector=selector), tempfile.TemporaryDirectory() as directory:
                artifacts = Path(directory) / "artifacts"
                artifacts.mkdir()
                attack = self.make_attack(artifacts)
                attack.options = SimpleNamespace(protocol=SELECTOR_PROTOCOL,
                    selector=selector, local_graph="pre", global_transition="attribute")
                attack.evaluations = {}
                adj = sp.csr_matrix([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
                labels, targets = np.array([0, 1, 0]), np.array([1, 2])
                with patch.object(Heirattack, "inner_train", return_value=None):
                    attack.meta_attack_multi_step(np.ones((3, 2), dtype=np.float32),
                        adj, labels, np.array([0]), targets, 1)
                output = np.array([[1., 0.], [0., 1.], [1., 0.]])
                attack.record_victim_evaluation("Clean graph", output, labels, targets)
                attack.record_victim_evaluation("Combined attack", output, labels, targets)
                metadata, evaluation = load_completed_artifacts(Path(directory), SELECTOR_PROTOCOL)
                self.assertEqual(metadata["protocol"], SELECTOR_PROTOCOL)
                self.assertEqual(evaluation["protocol"], SELECTOR_PROTOCOL)
                self.assertEqual(metadata["local_graph"], "pre")
                self.assertEqual(metadata["requested_local_graph"], "pre")
                self.assertEqual(attack.diagnostics[0]["local_edge_value"], 0)
                self.assertEqual(attack.diagnostics[0]["candidate_universe"], 3)
                if selector == "pr":
                    self.assertEqual(metadata["pr_solver"], pr_solver_settings(.2))
                    for filename, record in (("attack.json", metadata), ("evaluation.json", evaluation)):
                        path = artifacts / filename
                        path.write_text(json.dumps({**record, "protocol": PROTOCOL}))
                        with self.assertRaises(ValueError):
                            load_completed_artifacts(Path(directory), SELECTOR_PROTOCOL)
                        path.write_text(json.dumps(record))
                hashes.append(metadata["actions_hash"])
                with self.assertRaises(ValueError):
                    load_completed_artifacts(Path(directory))
        self.assertEqual(len(set(hashes)), 1)

    def test_real_loop_saves_actions_and_timing_without_changing_edge_order(self):
        for features_enabled, exhausted in ((True, False), (True, True), (False, False)):
            with self.subTest(features=features_enabled, exhausted=exhausted), \
                    tempfile.TemporaryDirectory() as directory:
                attack = self.make_attack(directory, features_enabled=features_enabled,
                                          exhausted=exhausted)
                adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                                            dtype=np.float32))
                original = adj.copy()
                seen_edges = []
                select = attack.select_text_candidate_nodes

                def capture_selection(fea, adj, row_idx, col_idx):
                    seen_edges.append(int(adj[row_idx, col_idx]))
                    return select(fea, adj, row_idx, col_idx)

                attack.select_text_candidate_nodes = capture_selection
                # Only neural training is stubbed; the real attack loop, global
                # scores, selector, edge flip and artifact writer all execute.
                with patch.object(Heirattack, "inner_train", return_value=None):
                    attack.meta_attack_multi_step(
                        np.ones((3, 2), dtype=np.float32), adj, np.array([0, 1, 0]),
                        np.array([0]), np.array([1, 2]), 1)

                metadata = json.loads((Path(directory) / "attack.json").read_text())
                rounds = json.loads((Path(directory) / "rounds.json").read_text())["rounds"]
                self.assertTrue(metadata["budget_complete"])
                self.assertEqual(metadata["actions"], [[0, 2, 1]])
                self.assertGreaterEqual(metadata["structural_seconds"], 0)
                self.assertEqual(metadata["structural_seconds"], rounds[0]["structural_seconds"])
                self.assertEqual(rounds[0]["local_edge_value"], 0)
                self.assertEqual(rounds[0]["local_graph"], "pre")
                self.assertEqual(metadata["local_graph"], "pre")
                self.assertEqual(metadata["requested_local_graph"], "post")
                self.assertEqual(seen_edges, [0] if features_enabled else [])
                saved = sp.load_npz(Path(directory) / "modified_adj.npz")
                self.assertEqual(saved[0, 2], 1)
                self.assertEqual((saved != original).nnz, 2)
                self.assertEqual((adj != original).nnz, 0)


if __name__ == "__main__":
    unittest.main()
