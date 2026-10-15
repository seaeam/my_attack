"""Synthetic artifact audit and paired summaries; no training or network calls."""
import contextlib
import copy
import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

from gbhaa_experiments.action_state_protocol import (
    ARMS, PROTOCOL, actions_hash, feature_measures, flip_action,
    independent_restarts, make_jobs, operator_settings, protocol_settings,
)
from gbhaa_experiments.followup_protocol import array_hash, graph_hash, prediction_metrics, seed_stream
from gbhaa_experiments.run_followup import fingerprint


class ActionStateSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "results"
        self.output = Path(self.temp.name) / "reports"

    @staticmethod
    def provenance():
        digest = hashlib.sha256(b"historical synthetic input").hexdigest()
        return dict(protocol=PROTOCOL, sources={"old.py": digest},
                    input_files={"old.npz": digest}, python_executable="/old/python",
                    python_environment={"python": "fixture"}, api_base_url="http://offline.invalid",
                    generation_model="fixture", machine="fixture", cuda_visible_devices=None,
                    thread_count=1, generation_temperature=0.)

    def directory(self, row):
        return self.root / row["dataset"] / f"ptb_{row['budget']:g}" / f"seed_{row['seed']}" / row["variant"]

    @staticmethod
    def job(row):
        return {k: row[k] for k in ("dataset", "seed", "budget", "variant")}

    @staticmethod
    def counts(writes=4):
        return dict(targets=writes, feature_writes=writes, feature_changes=0,
                    template_attempts=1, template_successes=1, template_failures=0,
                    node_failures=0, cluster_failures=0, cache_hits=0)

    def save_record(self, row):
        run = self.directory(row)
        for name in ("trajectory",) if row["variant"] == "structure" else ("attack", "evaluation"):
            (run / "artifacts" / f"{name}.json").write_text(json.dumps(row[name]))
        (run / "result.json").write_text(json.dumps(row))

    def refresh_fingerprint(self, row):
        row["fingerprint"] = fingerprint(dict(job=self.job(row), settings=row["settings"],
            provenance=row["provenance"], trajectory_hash=None if row["variant"] == "structure"
            else row["attack"]["trajectory_hash"]))

    def save_cache(self, row, *, request_changes=None, response="合成摘要 ||| Second template"):
        request = {"messages": [{"role": "system", "content": "Generate templates."},
                    {"role": "user", "content": "合成请求"}],
                   "model": row["attack"]["generation_model"],
                   "seed": row["attack"]["generation_seed"], "temperature": 0., "max_tokens": 50}
        request.update(request_changes or {})
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        path = self.root / "_generation_cache" / fingerprint(row["provenance"]) / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(key=key, request=request, response=response,
                       response_hash=hashlib.sha256(response.encode()).hexdigest())
        path.write_text(json.dumps(payload, ensure_ascii=False))
        row["attack"]["generation_request_keys"] = [key]
        row["attack"]["request_cache"] = {"hits": 1, "misses": 1}
        self.save_record(row)
        return path

    def set_predictions(self, row, correct):
        artifacts = self.directory(row) / "artifacts"
        ids = np.arange(6, dtype=np.int64)
        labels = np.zeros(6, dtype=np.int64)
        clean = np.zeros(6, dtype=np.int64)
        adv = np.array([0] * correct + [1] * (6 - correct), dtype=np.int64)
        for name, pred in (("clean", clean), ("combined", adv)):
            probabilities = np.full((6, 2), .1)
            probabilities[np.arange(6), pred] = .9
            np.savez(artifacts / f"{name}_predictions.npz", node_ids=ids, labels=labels,
                     predictions=pred, log_probabilities=np.log(probabilities))
        metrics = prediction_metrics(clean, adv, labels)
        metrics.update(combined_accuracy=metrics["attacked_accuracy"], edge_accuracy=1., feature_accuracy=1.)
        row["evaluation"] = dict(protocol=PROTOCOL, victim_seed=seed_stream(row["seed"], "victim"),
                                 clean_predictions_hash=array_hash(clean), metrics=metrics)

    def fixture(self, seeds=(15,), budget=.05, dataset="citeseer"):
        dense = np.zeros((12, 12), dtype=float)
        for u, v in list(zip(*np.triu_indices(12, 1)))[:40]:
            dense[u, v] = dense[v, u] = 1
        original = sp.csr_matrix(dense)
        features = np.eye(12, dtype=np.float32)
        target = int(40 * budget)
        actions = [[0, 1, 0], [10, 11, 1]]
        actions += [[0, i, 0] for i in range(2, target)]
        actions = actions[:target]
        modified = original.copy()
        for action in actions:
            modified = flip_action(modified, action)
        rows = []
        for job in make_jobs([dataset], seeds, [budget]):
            artifacts = self.directory(job) / "artifacts"
            artifacts.mkdir(parents=True)
            sp.save_npz(artifacts / "original_adj.npz", original)
            sp.save_npz(artifacts / "modified_adj.npz", modified)
            metadata = dict(**job, protocol=PROTOCOL, settings=protocol_settings(),
                operator_settings=operator_settings(), original_adj_hash=graph_hash(original),
                original_features_hash=array_hash(features), training_nodes_hash=array_hash(np.arange(6, 12)),
                target_nodes_hash=array_hash(np.arange(6)), evaluation_labels_hash=array_hash(np.zeros(6, dtype=np.int64)),
                target_edges=target, completed_edges=target, actions=actions, actions_hash=actions_hash(actions),
                modified_adj_hash=graph_hash(modified), budget_complete=True, structural_seconds=.5)
            row = dict(**job, protocol=PROTOCOL, settings=protocol_settings(), status="complete", returncode=0,
                       provenance=self.provenance(), command=["/old/python", "-m", "fixture"],
                       wall_time_seconds=2., peak_rss_mib=100.)
            rounds = []
            state = original.copy()
            for i, action in enumerate(actions):
                post = flip_action(state, action)
                diagnostic = dict(round=i, action=action, pre_graph_hash=graph_hash(state), post_graph_hash=graph_hash(post))
                if job["variant"] != "structure":
                    local = "post" if job["variant"] == "endpoint_post" else "pre"
                    restarts = independent_restarts(12, job["seed"], i) if job["variant"] == "parallel" else action[:2]
                    diagnostic.update(local_graph=local, diffusion_graph_hash=graph_hash(post if local == "post" else state),
                        local_edge_value=action[2] if local == "post" else 1-action[2], restart_nodes=restarts,
                        selected_nodes=[0, 1, 2, 3], candidate_universe=12, eligible_count=12,
                        current_features_hash=array_hash(features), probe_pre_nodes=[0, 1, 2, 3],
                        probe_post_nodes=[0, 1, 2, 3], probe_jaccard=1., probe_score_l1=0.,
                        action_similarity=0., attribute_counts=self.counts(), grouping_graph="post",
                        grouping_graph_hash=graph_hash(post))
                    if job["variant"] == "parallel":
                        diagnostic["independent_restart_overlap"] = len(set(restarts) & set(action[:2]))
                rounds.append(diagnostic)
                state = post
            (artifacts / "rounds.json").write_text(json.dumps({"rounds": rounds}))
            if job["variant"] == "structure":
                row["trajectory"] = metadata
            else:
                counts = {k: v * target for k, v in self.counts().items()}
                metadata.update(trajectory_hash=actions_hash(actions), selector="ae_pr", global_source="degree",
                    local_graph="post" if job["variant"] == "endpoint_post" else "pre",
                    generation_seed=seed_stream(job["seed"], "generation"), generation_temperature=0., generation_model="fixture",
                    vocabulary_fallback=False, vocabulary_partial=False, **feature_measures(features, features),
                    modified_features_hash=array_hash(features), attribute_node_visits=4*target,
                    visited_attribute_nodes=4, attribute_counts=counts, request_cache={"hits": 0, "misses": 0},
                    generation_request_keys=[])
                row["attack"] = metadata
                np.savez(artifacts / "features.npz", original=features, modified=features)
                np.savez(artifacts / "feature_updates.npz", nodes=np.tile(np.arange(4), (target, 1)),
                         values=np.tile(features[:4][None, :, :], (target, 1, 1)))
                self.set_predictions(row, {"parallel": 5, "endpoint_pre": 4, "endpoint_post": 2}[job["variant"]])
            self.refresh_fingerprint(row)
            self.save_record(row)
            rows.append(row)
        return rows

    def validate(self, row):
        module = importlib.import_module("gbhaa_experiments.action_state_validation")
        return module.validate_result(self.directory(row) / "result.json", self.job(row))

    def run_summary(self, seeds=(15,), budgets=(.05,), datasets=("citeseer",)):
        module = importlib.import_module("gbhaa_experiments.summarize_action_state")
        with contextlib.redirect_stdout(io.StringIO()):
            code = module.main(["--root", str(self.root), "--output-dir", str(self.output),
                "--datasets", *datasets, "--seeds", *map(str, seeds), "--budgets", *map(str, budgets)])
        return code, json.loads((self.output / "summary.json").read_text())

    def test_valid_historical_artifacts_and_successful_noop_writes_are_accepted_readonly(self):
        rows = self.fixture()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        for row in rows:
            self.assertEqual(self.validate(row), row)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_load_stage_before_result_completion(self):
        rows = self.fixture()
        module = importlib.import_module("gbhaa_experiments.action_state_validation")
        for row in rows:
            (self.directory(row) / "result.json").unlink()
            stage = module.load_stage(self.directory(row), self.job(row))
            self.assertIn("trajectory" if row["variant"] == "structure" else "evaluation", stage)

    def test_swapped_seed_and_copied_arm_are_rejected(self):
        rows = self.fixture(seeds=(15, 16))
        target = next(r for r in rows if r["seed"] == 16 and r["variant"] == "endpoint_post")
        source = next(r for r in rows if r["seed"] == 15 and r["variant"] == "endpoint_post")
        shutil.copytree(self.directory(source) / "artifacts", self.directory(target) / "artifacts", dirs_exist_ok=True)
        with self.assertRaisesRegex(ValueError, "seed"):
            self.validate(target)
        target = next(r for r in rows if r["seed"] == 15 and r["variant"] == "endpoint_pre")
        shutil.copytree(self.directory(source) / "artifacts", self.directory(target) / "artifacts", dirs_exist_ok=True)
        with self.assertRaisesRegex(ValueError, "variant"):
            self.validate(target)

    def test_fingerprint_and_provenance_fields_are_validated(self):
        base = self.fixture()[1]
        for key, value in (("fingerprint", "0"*64), ("returncode", 1), ("wall_time_seconds", -1)):
            row = copy.deepcopy(base); row[key] = value; self.save_record(row)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.validate(row)
        for key, value in (("thread_count", 2), ("sources", {}), ("generation_temperature", .2)):
            row = copy.deepcopy(base); row["provenance"][key] = value
            self.refresh_fingerprint(row); self.save_record(row)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.validate(row)

    def test_round_graph_restart_eligibility_and_counter_corruption_rejected(self):
        row = self.fixture()[1]
        path = self.directory(row) / "artifacts/rounds.json"
        original = json.loads(path.read_text())
        for field, value in (("diffusion_graph_hash", "0"*64), ("grouping_graph_hash", "0"*64),
                             ("local_edge_value", 0), ("restart_nodes", [0, 1]),
                             ("selected_nodes", [0, 0, 1, 2]), ("eligible_count", 11),
                             ("probe_jaccard", .5), ("current_features_hash", "0"*64)):
            changed = copy.deepcopy(original); changed["rounds"][0][field] = value
            path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.validate(row)
        changed = copy.deepcopy(original); changed["rounds"][0]["attribute_counts"]["feature_writes"] = 3
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "feature_writes"):
            self.validate(row)

    def test_wrong_final_graph_and_features_are_rejected(self):
        row = self.fixture()[1]
        artifacts = self.directory(row) / "artifacts"
        saved = (artifacts / "modified_adj.npz").read_bytes()
        shutil.copyfile(artifacts / "original_adj.npz", artifacts / "modified_adj.npz")
        with self.assertRaisesRegex(ValueError, "modified_adj_hash"):
            self.validate(row)
        (artifacts / "modified_adj.npz").write_bytes(saved)
        original = np.eye(12, dtype=np.float32); modified = original.copy(); modified[5, 0] = 1
        np.savez(artifacts / "features.npz", original=original, modified=modified)
        with self.assertRaisesRegex(ValueError, "modified_features_hash"):
            self.validate(row)

    def test_corrupt_predictions_and_false_accuracy_are_rejected(self):
        row = self.fixture()[1]
        path = self.directory(row) / "artifacts/combined_predictions.npz"
        with np.load(path) as data:
            content = {k: data[k] for k in data.files}
        changed = {**content, "log_probabilities": np.full((6, 2), np.nan)}
        np.savez(path, **changed)
        with self.assertRaisesRegex(ValueError, "log_probabilities"):
            self.validate(row)
        np.savez(path, **{**content, "predictions": 1-content["predictions"]})
        with self.assertRaisesRegex(ValueError, "predictions"):
            self.validate(row)
        np.savez(path, **content)
        row["evaluation"]["metrics"]["combined_accuracy"] = .123
        self.save_record(row)
        with self.assertRaisesRegex(ValueError, "combined_accuracy"):
            self.validate(row)

    def test_feature_updates_replay_checks_intermediate_and_final_features(self):
        row = self.fixture()[1]
        path = self.directory(row) / "artifacts/feature_updates.npz"
        with np.load(path) as data:
            nodes, values = data["nodes"], data["values"]
        for corruption in ("nonfinite", "intermediate", "final", "nodes"):
            changed_nodes, changed_values = nodes.copy(), values.copy()
            if corruption == "nonfinite":
                changed_values[0, 0, 0] = np.nan
            elif corruption == "nodes":
                changed_nodes[0, 0] = 8
            else:
                changed_values[0 if corruption == "intermediate" else -1, 0, 0] = 0
            np.savez(path, nodes=changed_nodes, values=changed_values)
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                self.validate(row)

    def test_real_feature_changes_are_replayed_and_norms_verified(self):
        row = self.fixture()[1]
        artifacts = self.directory(row) / "artifacts"
        original = np.eye(12, dtype=np.float32)
        modified = original.copy(); modified[0, 1] = .5
        updates = np.stack([modified[:4], modified[:4]])
        np.savez(artifacts / "feature_updates.npz", nodes=np.tile(np.arange(4), (2, 1)), values=updates)
        np.savez(artifacts / "features.npz", original=original, modified=modified)
        rounds_path = artifacts / "rounds.json"
        rounds = json.loads(rounds_path.read_text())
        rounds["rounds"][1]["current_features_hash"] = array_hash(modified)
        rounds["rounds"][0]["attribute_counts"]["feature_changes"] = 1
        rounds_path.write_text(json.dumps(rounds))
        row["attack"].update(modified_features_hash=array_hash(modified), **feature_measures(original, modified))
        row["attack"]["attribute_counts"]["feature_changes"] = 1
        self.save_record(row)
        self.assertEqual(self.validate(row)["attack"]["feature_l1"], .5)
        row["attack"]["feature_l1"] = .4; self.save_record(row)
        with self.assertRaisesRegex(ValueError, "feature_l1"):
            self.validate(row)

    def test_node_visit_limit_rejects_oversubscribed_round(self):
        row = self.fixture(budget=.3)[1]
        with self.assertRaisesRegex(ValueError, "eligible_count"):
            self.validate(row)

    def test_failed_writes_and_missing_artifact_are_rejected(self):
        row = self.fixture()[1]
        row["attack"]["attribute_counts"]["node_failures"] = 1
        self.save_record(row)
        with self.assertRaisesRegex(ValueError, "node_failures"):
            self.validate(row)
        row["attack"]["attribute_counts"]["node_failures"] = 0
        self.save_record(row)
        (self.directory(row) / "artifacts/feature_updates.npz").unlink()
        with self.assertRaisesRegex(ValueError, "feature_updates.npz"):
            self.validate(row)

    def test_block_rejects_mismatched_saved_provenance(self):
        row = self.fixture()[1]
        row["provenance"]["sources"]["old.py"] = "f"*64
        self.refresh_fingerprint(row); self.save_record(row)
        code, report = self.run_summary()
        self.assertEqual(code, 2)
        self.assertFalse(report["complete_blocks"])
        self.assertIn("provenance", json.dumps(report["rejected_blocks"]))

    def test_request_counters_must_be_compatible_with_distinct_request_keys(self):
        row = self.fixture()[1]
        row["attack"]["request_cache"] = {"hits": 0, "misses": 100}
        self.save_record(row)
        with self.assertRaisesRegex(ValueError, "generation_request_keys"):
            self.validate(row)

    def test_referenced_unicode_request_cache_is_validated_readonly(self):
        row = self.fixture()[1]
        self.save_cache(row)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.validate(row), row)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_missing_or_corrupt_referenced_cache_is_rejected(self):
        row = self.fixture()[1]
        path = self.save_cache(row)
        saved = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "generation cache"):
            self.validate(row)
        path.write_bytes(saved)
        payload = json.loads(saved)
        for key, value in (("key", "0"*64), ("response_hash", "0"*64),
                           ("request", {**payload["request"], "messages": [{"role": "user", "content": "altered"}]})):
            changed = {**payload, key: value}
            path.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "generation cache"):
                self.validate(row)
        path.write_text("{broken")
        with self.assertRaisesRegex(ValueError, "generation cache"):
            self.validate(row)

    def test_cache_settings_must_match_recorded_arm_even_with_consistent_hashes(self):
        row = self.fixture()[1]
        for key, value in (("model", "other-model"), ("seed", 99), ("temperature", .2),
                           ("max_tokens", 100), ("max_tokens", 50.), ("messages", [])):
            self.save_cache(row, request_changes={key: value})
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                self.validate(row)

    def test_empty_parsed_templates_rejected_even_with_matching_response_hash(self):
        row = self.fixture()[1]
        for response in ("", "   ", " ||| ||| "):
            self.save_cache(row, response=response)
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "templates"):
                self.validate(row)

    def test_action_similarity_is_recomputed_from_current_features(self):
        row = self.fixture()[1]
        path = self.directory(row) / "artifacts/rounds.json"
        rounds = json.loads(path.read_text())
        rounds["rounds"][0]["action_similarity"] = .5
        path.write_text(json.dumps(rounds))
        with self.assertRaisesRegex(ValueError, "action_similarity"):
            self.validate(row)

    def test_cli_rejects_duplicate_axes_and_invalid_budgets(self):
        for kwargs in (dict(seeds=(15, 15)), dict(datasets=("cora", "cora")),
                       dict(budgets=(0,)), dict(budgets=(float("nan"),)),
                       dict(budgets=(.05, .0500000001))):
            with self.subTest(kwargs=kwargs), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.run_summary(**kwargs)
                self.assertEqual(raised.exception.code, 2)

    def test_missing_arm_excludes_whole_seed(self):
        rows = self.fixture(seeds=(15, 16))
        missing = next(r for r in rows if r["seed"] == 16 and r["variant"] == "endpoint_pre")
        (self.directory(missing) / "result.json").unlink()
        code, report = self.run_summary(seeds=(15, 16))
        self.assertEqual(code, 2)
        self.assertEqual(len(report["complete_blocks"]), 1)
        self.assertEqual(len(report["missing_jobs"]), 1)
        self.assertTrue(all(r["n"] == 1 and r["sd"] is None and r["ci95"] is None for r in report["summary"]))
        self.assertTrue(all(r["preliminary"] for r in report["summary"]))
        with (self.output / "per_seed.csv").open() as handle:
            exported = list(csv.DictReader(handle))
        self.assertEqual(len(exported), 3)
        self.assertIn("feature_l1", exported[0])

    def test_hand_computed_paired_differences_keep_requested_seed_order(self):
        rows = self.fixture(seeds=(15, 16))
        correct = {15: {"parallel": 6, "endpoint_pre": 4, "endpoint_post": 2},
                   16: {"parallel": 5, "endpoint_pre": 3, "endpoint_post": 2}}
        for row in rows:
            if row["variant"] in ARMS:
                self.set_predictions(row, correct[row["seed"]][row["variant"]]); self.save_record(row)
        code, report = self.run_summary(seeds=(16, 15))
        self.assertEqual(code, 0)
        primary = next(r for r in report["contrasts"] if r["contrast"] == "pre_minus_post")
        self.assertEqual(primary["seeds"], [16, 15])
        self.assertAlmostEqual(primary["mean"], 25)
        self.assertAlmostEqual(primary["sd"], 100/6/np.sqrt(2))
        self.assertEqual(primary["role"], "primary")
        self.assertEqual(len(report["paired_differences"]), 6)
        self.assertTrue((self.output / "paired_differences.csv").exists())

    def test_mixed_provenance_suppresses_cross_seed_aggregates(self):
        rows = self.fixture(seeds=(15, 16))
        for row in rows:
            if row["seed"] == 16:
                row["provenance"]["machine"] = "other"
                self.refresh_fingerprint(row); self.save_record(row)
        code, report = self.run_summary(seeds=(15, 16))
        self.assertEqual(code, 2)
        self.assertFalse(report["summary"])
        self.assertFalse(report["contrasts"])
        self.assertIn("mixed provenance", " ".join(report["warnings"]))

    def test_primary_holm_family_retains_missing_requested_cells(self):
        self.fixture(seeds=(15, 16, 17, 18, 19))
        code, report = self.run_summary(seeds=(15, 16, 17, 18, 19), budgets=(.05, .1), datasets=("citeseer", "cora"))
        self.assertEqual(code, 2)
        self.assertEqual(report["holm_family"]["size"], 4)
        self.assertEqual(len(report["holm_family"]["missing_contrasts"]), 3)
        primary = next(r for r in report["contrasts"] if r["role"] == "primary")
        self.assertAlmostEqual(primary["p_signflip"], .0625)
        self.assertAlmostEqual(primary["p_holm"], .25)
        self.assertTrue(all(r["p_holm"] is None for r in report["contrasts"] if r["role"] == "secondary"))


if __name__ == "__main__":
    unittest.main()
