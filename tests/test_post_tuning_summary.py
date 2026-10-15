"""Synthetic Post tuning audits; no training, held-out evaluation or network calls."""
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

import test_action_state_summary as historical
from gbhaa_experiments.action_state_protocol import actions_hash, flip_action
from gbhaa_experiments.followup_protocol import array_hash, graph_hash, seed_stream
from gbhaa_experiments.run_followup import fingerprint


class PostTuningModuleTests(unittest.TestCase):
    def test_validator_and_summary_entrypoints_exist(self):
        self.assertIsNotNone(importlib.util.find_spec("gbhaa_experiments.post_tuning_validation"))
        self.assertIsNotNone(importlib.util.find_spec("gbhaa_experiments.summarize_post_tuning"))


class PostTuningSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "tuning"
        self.output = Path(self.temp.name) / "reports"

    @staticmethod
    def protocol():
        return importlib.import_module("gbhaa_experiments.post_tuning_protocol")

    @staticmethod
    def job(row):
        return {key: row[key] for key in ("dataset", "seed", "source_budget", "max_actions", "trial_id", "arm", "profile_id", "config")}

    def directory(self, job):
        return self.root / job["dataset"] / f"source_ptb_{job['source_budget']:g}" / f"steps_{job['max_actions']}" / f"seed_{job['seed']}" / job["trial_id"]

    def save(self, row):
        run = self.directory(row)
        for name in ("attack", "evaluation"):
            (run / "artifacts" / f"{name}.json").write_text(json.dumps(row[name]))
        row["fingerprint"] = fingerprint(dict(job=self.job(row), provenance=row["provenance"], parent_fingerprint=row["parent_fingerprint"]))
        (run / "result.json").write_text(json.dumps(row))

    def predictions(self, row, correct=1, confidence=.9):
        artifacts = self.directory(row) / "artifacts"
        ids = np.arange(3, dtype=np.int64)
        labels = np.zeros(3, dtype=np.int64)
        clean = np.zeros(3, dtype=np.int64)
        adv = np.array([0]*correct + [1]*(3-correct), dtype=np.int64)
        losses = []
        for name, pred, conf in (("clean_validation", clean, .9), ("validation", adv, confidence)):
            probs = np.full((3, 2), 1-conf)
            probs[np.arange(3), pred] = conf
            logits = np.log(probs)
            losses.append(float(-logits[:, 0].mean()))
            np.savez(artifacts / f"{name}_predictions.npz", node_ids=ids, labels=labels, predictions=pred, log_probabilities=logits)
        row["evaluation"] = dict(protocol=self.protocol().PROTOCOL, split="validation", victim_seed=seed_stream(row["seed"], "victim"),
            validation_nodes_hash=array_hash(ids), holdout_nodes_hash=row["attack"]["holdout_nodes_hash"],
            train_nodes_hash=row["attack"]["training_nodes_hash"], clean_predictions_hash=array_hash(clean),
            validation_predictions_hash=array_hash(adv), target_count=3,
            metrics=dict(clean_validation_accuracy=1., validation_accuracy=correct/3,
                clean_validation_nll=losses[0], validation_nll=losses[1], accuracy_drop_pp=100*(1-correct/3)))

    def fixture(self, seeds=(15,), profiles=("baseline", "alpha010"), matched=False):
        p = self.protocol()
        old = historical.ActionStateSummaryTests()
        old.setUp()
        self.addCleanup(old.doCleanups)
        parents = old.fixture(seeds=seeds, budget=.1)
        by_key = {(r["seed"], r["variant"]): r for r in parents}
        jobs = p.make_jobs(["citeseer"], list(seeds), .1, 2, profile_ids=list(profiles), matched_controls=matched)
        rows = []
        for job in jobs:
            ancestor = by_key[job["seed"], "structure"]
            source = by_key[job["seed"], job["arm"]]
            artifacts = self.directory(job) / "artifacts"
            shutil.copytree(old.directory(source) / "artifacts", artifacts)
            for name in ("clean_predictions.npz", "combined_predictions.npz"):
                (artifacts / name).unlink()
            original = sp.load_npz(artifacts / "original_adj.npz")
            actions = ancestor["trajectory"]["actions"][:job["max_actions"]]
            modified = original.copy()
            for action in actions:
                modified = flip_action(modified, action)
            sp.save_npz(artifacts / "modified_adj.npz", modified)
            rounds = json.loads((artifacts / "rounds.json").read_text())["rounds"][:len(actions)]
            (artifacts / "rounds.json").write_text(json.dumps({"rounds": rounds}))
            with np.load(artifacts / "feature_updates.npz") as data:
                nodes, values = data["nodes"][:len(actions)], data["values"][:len(actions)]
            np.savez(artifacts / "feature_updates.npz", nodes=nodes, values=values)
            np.savez(artifacts / "splits.npz", train=np.arange(6,12), validation=np.arange(3), test=np.arange(3,6))
            attack = copy.deepcopy(source["attack"])
            attack.update(protocol=p.PROTOCOL, variant=job["arm"], profile_id=job["profile_id"], trial_id=job["trial_id"], config=job["config"],
                settings=p.runtime_settings(job["config"]), operator_settings=p.operator_settings(job["config"]),
                source_budget=job["source_budget"], budget=len(actions)/40, effective_budget=len(actions)/40,
                prefix_length=len(actions), target_edges=len(actions), completed_edges=len(actions),
                actions=actions, actions_hash=actions_hash(actions), trajectory_hash=actions_hash(actions), modified_adj_hash=graph_hash(modified),
                parent_protocol=ancestor["protocol"], parent_fingerprint=ancestor["fingerprint"], parent_actions_hash=ancestor["trajectory"]["actions_hash"],
                validation_nodes_hash=array_hash(np.arange(3)), holdout_nodes_hash=array_hash(np.arange(3,6)),
                generation_max_tokens=job["config"]["generation_max_tokens"], attribute_node_visits=4*len(actions),
                attribute_counts={key: value*len(actions) for key,value in old.counts().items()})
            provenance = {**old.provenance(), "protocol":p.PROTOCOL}
            row = dict(**job, protocol=p.PROTOCOL, settings=p.runtime_settings(job["config"]), provenance=provenance,
                parent_run=str(old.directory(ancestor).resolve()), parent_fingerprint=ancestor["fingerprint"],
                status="complete", returncode=0, command=["fixture"], wall_time_seconds=2., peak_rss_mib=100., attack=attack)
            self.predictions(row, correct=2 if job["profile_id"] == "baseline" else 1)
            self.save(row)
            rows.append(row)
        (self.root / "plan.json").write_text(json.dumps(dict(protocol=p.PROTOCOL,jobs=jobs,provenance=rows[0]["provenance"])))
        return rows

    def validate(self, row, job=None):
        return importlib.import_module("gbhaa_experiments.post_tuning_validation").validate_result(self.directory(row)/"result.json", job or self.job(row))

    def summarize(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = importlib.import_module("gbhaa_experiments.summarize_post_tuning").main(["--root",str(self.root),"--output-dir",str(self.output)])
        return code, json.loads((self.output / "summary.json").read_text()), json.loads((self.output / "recommendation.json").read_text())

    def test_matched_guided_profiles_require_same_fixed_surrogate(self):
        from gbhaa_experiments.summarize_post_tuning import _block_errors
        rows=self.fixture(profiles=('frontier_editgain',),matched=True)
        guided=[r for r in rows if r['profile_id']=='frontier_editgain']
        for r in guided:r['attack']['selector_surrogate']={'identity':'same'}
        paths=[self.directory(r)/'result.json' for r in rows]
        self.assertEqual(_block_errors(rows,paths),[])
        guided[-1]['attack']['selector_surrogate']={'identity':'different'}
        self.assertIn('mismatched fixed selector surrogate/pseudo-labels/targets',_block_errors(rows,paths))

    def test_coverage_rules_are_included_in_surrogate_identity_audit(self):
        from gbhaa_experiments.summarize_post_tuning import _block_errors
        rows=self.fixture(profiles=('frontier_editgain','frontier_editgain_decay','frontier_editgain_fresh3'),matched=True)
        guided=[r for r in rows if r['profile_id']!='baseline']
        for r in guided:r['attack']['selector_surrogate']={'identity':'same'}
        paths=[self.directory(r)/'result.json' for r in rows]
        self.assertEqual(_block_errors(rows,paths),[])
        guided[-1]['attack']['selector_surrogate']={'identity':'different'}
        self.assertIn('mismatched fixed selector surrogate/pseudo-labels/targets',_block_errors(rows,paths))

    def test_complete_historical_parent_prefix_and_plan_summary(self):
        rows = self.fixture(seeds=(15,16))
        for row in rows:
            self.assertEqual(self.validate(row), row)
        code, report, selection = self.summarize()
        self.assertEqual(code, 0)
        self.assertEqual(selection["status"], "complete")
        self.assertEqual(selection["selection_split"], "validation")
        self.assertEqual(selection["recommendations"][0]["profile_id"], "alpha010")
        self.assertEqual(len(report["per_trial"]), len(rows))
        self.assertTrue((self.output / "ranking.csv").exists())
        self.assertTrue(report["comparisons"])
        self.assertTrue(all('probe_changed_set_fraction' in r for r in report['per_trial']))
        self.assertTrue((self.output / "paired_comparisons.csv").exists())
        self.assertTrue(all(r['expected_n'] == 2 and r['status'] == 'complete'
                            for r in report['comparisons']))
        self.assertIn('ACC-only W/L/T', (self.output / 'summary.md').read_text())

    def test_missing_control_seed_is_visible_in_comparison_export(self):
        rows = self.fixture(seeds=(15,16), profiles=('baseline','alpha010'), matched=True)
        missing = next(row for row in rows if row['seed'] == 16 and
                       row['profile_id'] == 'alpha010' and row['arm'] == 'endpoint_pre')
        (self.directory(missing) / 'result.json').unlink()
        code, report, selection = self.summarize()
        self.assertEqual(code, 2)
        row = next(r for r in report['comparisons'] if r['profile_id'] == 'alpha010'
                   and r['control_arm'] == 'endpoint_pre')
        self.assertEqual(row['expected_n'], 2)
        self.assertEqual(row['n'], 1)
        self.assertEqual(row['status'], 'partial')
        self.assertEqual(row['matrix_status'], 'incomplete')
        self.assertIsNone(row['control_minus_post_accuracy_pp_ci95'])
        self.assertEqual(selection['recommendations'], [])

    def test_incomplete_suppresses_recommendation_and_keeps_missing_row(self):
        rows = self.fixture()
        (self.directory(rows[-1]) / "result.json").unlink()
        code, report, selection = self.summarize()
        self.assertEqual(code, 2)
        self.assertEqual(selection["status"], "incomplete")
        self.assertEqual(selection["recommendations"], [])
        self.assertEqual(len(report["per_trial"]), len(rows))
        self.assertIn("missing", [r["status"] for r in report["per_trial"]])

    def test_metric_recomputation_and_log_probability_normalization(self):
        row = self.fixture()[0]
        row["evaluation"]["metrics"]["validation_accuracy"] = .123
        self.save(row)
        with self.assertRaisesRegex(ValueError,"validation_accuracy"):
            self.validate(row)
        self.predictions(row); self.save(row)
        path = self.directory(row) / "artifacts/validation_predictions.npz"
        with np.load(path) as data:
            values = {key:data[key] for key in data.files}
        values["log_probabilities"] += 1
        np.savez(path, **values)
        with self.assertRaisesRegex(ValueError,"normali|probabilit"):
            self.validate(row)

    def test_tiny_metric_falsification_cannot_change_the_tie_break(self):
        row = self.fixture()[0]
        row["evaluation"]["metrics"]["validation_nll"] += 1e-8
        self.save(row)
        with self.assertRaisesRegex(ValueError, "validation_nll"):
            self.validate(row)

    def test_heldout_metrics_are_rejected(self):
        row = self.fixture()[0]
        row["evaluation"]["test_accuracy"] = 0.
        self.save(row)
        with self.assertRaisesRegex(ValueError, "validation-only"):
            self.validate(row)

    def test_split_disjointness_and_validation_only_predictions(self):
        row = self.fixture()[0]
        artifacts = self.directory(row)/"artifacts"
        # Arbitrary, unreadable held-out prediction files never enter the audit or rank.
        (artifacts/"test_predictions.npz").write_bytes(b"intentionally unreadable")
        self.validate(row)
        np.savez(artifacts/"splits.npz", train=np.arange(6,12), validation=np.arange(3), test=np.arange(2,6))
        with self.assertRaisesRegex(ValueError,"disjoint|overlap"):
            self.validate(row)

    def test_prefix_config_parent_and_resume_identity_are_strict(self):
        base = self.fixture()[0]
        for field,value in (("actions", list(reversed(base["attack"]["actions"]))), ("parent_fingerprint","0"*64),
                            ("generation_max_tokens",51), ("operator_settings",{})):
            row=copy.deepcopy(base); row["attack"][field]=value; self.save(row)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(row)
        self.save(base)
        job=self.job(base); job["config"]={**job["config"],"alpha":.99}
        with self.assertRaisesRegex(ValueError,"config|profile|job"):
            self.validate(base,job)

    def test_cache_request_setting_must_match_actual_profile(self):
        row=self.fixture()[0]
        request=dict(messages=[dict(role="user",content="synthetic")], model="fixture",
            seed=row["attack"]["generation_seed"], temperature=0., max_tokens=row["config"]["generation_max_tokens"]+1)
        key=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        cache=self.root/"_generation_cache"/fingerprint(row["provenance"])/f"{key}.json"
        cache.parent.mkdir(parents=True)
        cache.write_text(json.dumps(dict(key=key, request=request,response="synthetic",response_hash=hashlib.sha256(b"synthetic").hexdigest())))
        row["attack"]["request_cache"]={"hits":0,"misses":1}
        row["attack"]["generation_request_keys"]=[key]
        self.save(row)
        with self.assertRaisesRegex(ValueError,"max_tokens"):
            self.validate(row)

    def test_validation_nll_then_stable_profile_id_breaks_accuracy_ties(self):
        rows=self.fixture(profiles=("baseline","alpha010","alpha015"))
        for row in rows:
            self.predictions(row,correct=1,confidence=.99 if row["profile_id"] != "baseline" else .9)
            self.save(row)
        code,report,selection=self.summarize()
        self.assertEqual(code,0)
        self.assertEqual(selection["recommendations"][0]["profile_id"],"alpha010")
        self.assertEqual([r["profile_id"] for r in report["ranking"]],["alpha010","alpha015","baseline"])
        self.assertIsNone(report["ranking"][0]["validation_accuracy_percent_sd"])

    def test_mixed_clean_predictions_or_provenance_reject_recommendation(self):
        rows=self.fixture()
        row=rows[-1]
        row["provenance"]["machine"]="different-machine"
        self.save(row)
        code,report,selection=self.summarize()
        self.assertEqual(code,2)
        self.assertEqual(selection["recommendations"],[])
        self.assertTrue(report["rejected_blocks"])
        self.assertTrue(all(r['status'] == 'rejected' for r in report['comparisons']))

    def test_rejected_matched_cell_has_no_inferential_interval_or_valid_seed_count(self):
        rows=self.fixture(seeds=(15,16), profiles=('baseline','alpha010'), matched=True)
        row=rows[-1]
        row['provenance']['machine']='different-machine'
        self.save(row)
        code,report,_=self.summarize()
        self.assertEqual(code,2)
        for comparison in report['comparisons']:
            self.assertEqual(comparison['status'],'rejected')
            self.assertEqual(comparison['n'],2)  # Observations retained for audit.
            self.assertIsNone(comparison['control_minus_post_accuracy_pp_ci95'])
            self.assertEqual(comparison['valid_n'],0)

    def test_partial_matrix_still_checks_available_rows_for_matching(self):
        rows=self.fixture(seeds=(15,16), profiles=('baseline','alpha010'), matched=True)
        # Both the mismatch and the missing run are in seed 16. The plan's
        # provenance remains unchanged; clean-output matching must still run.
        row=rows[-2]
        path=self.directory(row)/'artifacts/clean_validation_predictions.npz'
        with np.load(path) as data: values={key:data[key] for key in data.files}
        values['log_probabilities']=np.log(np.tile([.8,.2],(3,1)))
        np.savez(path,**values)
        row['evaluation']['metrics']['clean_validation_nll']=float(-np.log(.8))
        self.save(row)
        (self.directory(rows[-1])/'result.json').unlink()
        code,report,_=self.summarize()
        self.assertEqual(code,2)
        self.assertIn('clean validation outputs',str(report['rejected_blocks']))

    def test_matched_control_comparisons_keep_all_arms(self):
        rows = self.fixture(profiles=("alpha010",), matched=True)
        code,report,selection = self.summarize()
        self.assertEqual(code, 0)
        comparisons = [r for r in report["comparisons"] if r["profile_id"] == "alpha010" and r["control_arm"] != "endpoint_post"]
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(r["comparison_mode"] == "matched configuration" for r in comparisons))
        self.assertTrue(all(r["control_profile_id"] == "alpha010" for r in comparisons))
        self.assertTrue(all(r["n"] == 1 and r["control_minus_post_accuracy_pp_sd"] is None for r in comparisons))
        self.assertEqual(len(report["per_trial"]), len(rows))

    def test_clean_distribution_mismatch_suppresses_recommendation(self):
        rows = self.fixture()
        row = rows[-1]
        path = self.directory(row)/"artifacts/clean_validation_predictions.npz"
        with np.load(path) as data:
            values = {key:data[key] for key in data.files}
        values["log_probabilities"] = np.log(np.tile([.8,.2], (3,1)))
        np.savez(path, **values)
        row["evaluation"]["metrics"]["clean_validation_nll"] = float(-np.log(.8))
        self.save(row)
        self.validate(row)
        code,report,selection = self.summarize()
        self.assertEqual(code, 2)
        self.assertEqual(selection["recommendations"], [])
        self.assertIn("clean validation outputs", str(report["rejected_blocks"]))

    def test_saved_snapshot_and_failed_status_are_rejected(self):
        row = self.fixture()[0]
        path = self.directory(row)/"result.json"
        changed = copy.deepcopy(row)
        changed["attack"]["feature_l1"] = 1.
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "result.attack"):
            self.validate(row)
        changed = copy.deepcopy(row); changed["status"]="failed"; changed["returncode"]=1
        path.write_text(json.dumps(changed))
        code,report,selection = self.summarize()
        self.assertEqual(code, 2)
        self.assertEqual(selection["recommendations"], [])
        self.assertEqual(len(report["rejected_jobs"]), 1)

    def test_real_entrypoint_replay_and_evaluation_validate_and_summarize(self):
        import torch
        from heir import Heirattack
        from gbhaa_experiments.entrypoints.meta_post_tuning import TuningAttack, evaluate_validation

        class Victim:
            def __init__(self, **kwargs):
                pass
            def to(self, device):
                return self
            def fit(self, features, adj, labels, idx_train):
                self.output = torch.log(torch.tensor(np.tile([.8, .2], (len(features), 1)), dtype=torch.float32))

        p = self.protocol()
        original = historical.ActionStateSummaryTests()
        original.setUp(); self.addCleanup(original.doCleanups)
        parent = original.fixture()[0]
        source_run = original.directory(parent)
        with np.load(source_run.parent / "parallel/artifacts/features.npz") as data:
            features = data["original"]
        adj = sp.load_npz(source_run / "artifacts/original_adj.npz")
        jobs = p.make_jobs(["citeseer"], [15], .05, 1, profile_ids=["baseline", "floor025"], matched_controls=True)
        provenance = {**original.provenance(), "protocol":p.PROTOCOL}
        for job in jobs:
            cfg = job["config"]
            artifacts = self.directory(job) / "artifacts"
            attack = TuningAttack.__new__(TuningAttack); torch.nn.Module.__init__(attack)
            attack.config = cfg
            attack.args = SimpleNamespace(dataset="citeseer",seed=15,ptb_rate=.05, **p.runtime_settings(cfg))
            attack.options = SimpleNamespace(phase=job["arm"], artifacts=artifacts, source_run=source_run,
                max_actions=1, profile_id=job["profile_id"], trial_id=job["trial_id"])
            attack.trace_dir=artifacts; attack.nnodes=12; attack.nclass=2; attack.device="cpu"
            attack.freeze_structure_features=True; attack.use_oracle=False; attack.text_attack_total_nodes=None
            attack.text_generator=SimpleNamespace(model_name="fixture", max_tokens=cfg["generation_max_tokens"],api_stats={},
                uses_fallback_vocabulary=False, uses_partial_vocabulary=False)
            attack.request_cache=SimpleNamespace(stats={"hits":0,"misses":0},used_keys=set())
            attack.attribute_counts={}; attack.actions=[]; attack.diagnostics=[]; attack.evaluations={}
            attack.round_index=-1; attack.template_cache={}
            for key,value in p.runtime_settings(cfg).items():
                setattr(attack,key,value)
            attack.self_training_label=lambda y,idx:y
            attack.get_embeddings=lambda x,a:x
            attack._build_round_text_clusters=lambda nodes,*a,**kw:{0:nodes.tolist()}
            attack.configure_evaluation_split(np.arange(6,12),np.arange(3),np.arange(3,6))
            def writer(**kwargs):
                attack.finish_text_attack_round(dict(targets=4,feature_writes=4,feature_changes=0,
                    template_attempts=1,template_successes=1,template_failures=0,node_failures=0,
                    cluster_failures=0,cache_hits=0,completed=0))
                return kwargs["full_features"]
            attack.attack_features_with_text=writer
            with patch.object(Heirattack, "inner_train", return_value=None), contextlib.redirect_stdout(io.StringIO()):
                attack.meta_attack_multi_step(features,adj,np.zeros(12,dtype=np.int64),np.arange(6,12),np.arange(6),1)
                evaluation = evaluate_validation(attack,features,adj,attack.modified_features,attack.modified_adj,
                    np.zeros(12,dtype=np.int64),victim_class=Victim)
            stage = importlib.import_module("gbhaa_experiments.post_tuning_validation").load_trial(
                self.directory(job),job,parent_run=source_run)
            self.assertEqual(stage["evaluation"], evaluation)
            row=dict(**job,protocol=p.PROTOCOL,settings=p.runtime_settings(cfg),provenance=provenance,
                parent_run=str(source_run.resolve()),parent_fingerprint=parent["fingerprint"],status="complete",returncode=0,
                command=["synthetic-fixture"],wall_time_seconds=1.,peak_rss_mib=None,**stage)
            self.save(row)
            self.assertEqual(self.validate(row), row)
        (self.root/"plan.json").write_text(json.dumps(dict(protocol=p.PROTOCOL,jobs=jobs,provenance=provenance)))
        code,report,selection=self.summarize()
        self.assertEqual(code,0)
        self.assertEqual(len(report["per_trial"]),len(jobs))
        self.assertEqual(selection["recommendations"][0]["profile_id"],"baseline")


if __name__ == "__main__":
    unittest.main()
