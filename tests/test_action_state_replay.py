import contextlib
import io
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from heir import Heirattack
from gbhaa_experiments.entrypoints.meta_action_state import ActionStateAttack, StructureReady
from gbhaa_experiments.action_state_protocol import ARMS, PROTOCOL, actions_hash, protocol_settings, make_jobs
from gbhaa_experiments.followup_protocol import graph_hash, array_hash
from gbhaa_experiments.action_state_validation import load_stage, validate_result
from gbhaa_experiments.run_action_state import job_directory
from gbhaa_experiments.run_followup import fingerprint
from gbhaa_experiments.summarize_action_state import main as summarize


class ReplayTests(unittest.TestCase):
    def make_attack(self, directory, arm):
        attack = ActionStateAttack.__new__(ActionStateAttack)
        torch.nn.Module.__init__(attack)
        attack.args = SimpleNamespace(seed=15, dataset="synthetic", ptb_rate=.5,
                                      **protocol_settings())
        attack.options = SimpleNamespace(phase=arm, artifacts=Path(directory),
                                         trajectory=Path(directory)/'trajectory.json')
        attack.trace_dir = Path(directory)
        attack.nnodes=6; attack.nfeat=2; attack.nclass=2; attack.device='cpu'
        attack.freeze_structure_features=True; attack.use_oracle=False
        attack.text_generator=SimpleNamespace(model_name='offline', api_stats={},
            uses_fallback_vocabulary=False,uses_partial_vocabulary=False)
        attack.attribute_counts={};attack.diagnostics=[];attack.actions=[];attack.evaluations={}
        attack.round_index=-1;attack.template_cache={}
        for key,val in protocol_settings().items():setattr(attack,key,val)
        attack.text_attack_total_nodes=None
        attack.self_training_label=lambda labels, idx:labels
        attack.get_embeddings=lambda features, norm:features
        attack._build_round_text_clusters=lambda nodes,*a,**kw:{0:nodes.tolist()}
        attack.request_cache=SimpleNamespace(stats={'hits':0,'misses':0},used_keys=set())
        return attack

    def test_real_replay_uses_pre_or_post_and_executes_each_flip_once(self):
        adj=sp.csr_matrix(np.array([[0,1,0,0,0,0],[1,0,1,0,0,0],
            [0,1,0,1,0,0],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0]],dtype=float))
        acts=[[0,5,1],[2,3,0]]
        features=np.ones((6,2),dtype=np.float32);labels=np.array([0,1,0,1,0,1])
        final_hashes=[]
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                attack=self.make_attack(directory,arm)
                trajectory=attack.input_metadata(features,adj,labels,np.array([0]),np.arange(1,6),2)
                trajectory.update(variant='structure',actions=acts,actions_hash=actions_hash(acts))
                attack.options.trajectory.write_text(json.dumps(trajectory))
                seen=[]
                def writer(**kwargs):
                    seen.append(int(attack.full_adj.to_dense()[acts[len(seen)][0],acts[len(seen)][1]]))
                    attack.finish_text_attack_round(dict(targets=4,feature_writes=4,feature_changes=0,
                        template_attempts=1,template_successes=1,template_failures=0,
                        node_failures=0,cluster_failures=0,cache_hits=0,completed=0))
                    return kwargs['full_features']
                attack.attack_features_with_text=writer
                with patch.object(Heirattack,'inner_train',return_value=None),contextlib.redirect_stdout(io.StringIO()):
                    attack.meta_attack_multi_step(features,adj,labels,np.array([0]),np.arange(1,6),2)
                self.assertEqual(seen,[1,0])
                self.assertEqual(attack.actions,acts)
                self.assertEqual([d['local_edge_value'] for d in attack.diagnostics],
                                 [1,0] if arm=='endpoint_post' else [0,1])
                self.assertEqual(attack.metadata['attribute_node_visits'],8)
                self.assertEqual(attack.metadata['changed_feature_entries'],0)
                for row in attack.diagnostics:
                    self.assertEqual(row['grouping_graph'],'post')
                    self.assertEqual(len(row['selected_nodes']),4)
                final_hashes.append(graph_hash(attack.modified_adj))
        self.assertEqual(len(set(final_hashes)),1)

    def test_partial_writes_fail_but_four_successful_noops_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            attack=self.make_attack(directory,'endpoint_post')
            attack.diagnostics=[{}]
            stats=dict(targets=4,feature_writes=4,feature_changes=0,template_attempts=1,
                template_successes=1,template_failures=0,node_failures=0,cluster_failures=0,
                cache_hits=0,completed=0)
            attack.finish_text_attack_round(stats)
            self.assertEqual(attack.attribute_counts['feature_writes'],4)
            with self.assertRaises(ValueError):
                attack.finish_text_attack_round({**stats,'feature_writes':3,'node_failures':1})

    def test_real_writer_reports_partial_failure_and_accepts_successful_noops(self):
        for fail_node in (None,2):
            with self.subTest(fail_node=fail_node),tempfile.TemporaryDirectory() as directory:
                attack=self.make_attack(directory,'endpoint_post')
                attack.use_text_attack=True;attack.diagnostics=[{}]
                counter=[0]
                def vectorize(texts):
                    current=counter[0];counter[0]+=1
                    if current==fail_node:raise ValueError('synthetic vectorization failure')
                    return sp.csr_matrix([[1.,1.]])
                attack.text_generator=SimpleNamespace(
                    extract_words_from_bow_vector=lambda vector:(['alpha'],None),
                    generate_cluster_template=lambda **kw:['alpha'],
                    vectorizer=SimpleNamespace(transform=vectorize))
                attack._select_discriminative_words=lambda *a,**kw:[]
                features=torch.ones((6,2))
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    if fail_node is None:
                        result=attack.attack_features_with_text(np.arange(4),features,None,
                            text_clusters={0:[0,1,2,3]})
                        torch.testing.assert_close(result,features)
                        self.assertEqual(attack.attribute_counts['feature_writes'],4)
                        self.assertEqual(attack.attribute_counts['feature_changes'],0)
                    else:
                        with self.assertRaisesRegex(ValueError,'four successful writes'):
                            attack.attack_features_with_text(np.arange(4),features,None,
                                text_clusters={0:[0,1,2,3]})
                        self.assertEqual(attack.attribute_counts['feature_writes'],3)
                        self.assertEqual(attack.attribute_counts['node_failures'],1)

    def test_real_planner_replay_artifacts_pass_validation_and_summary(self):
        # Exercise the real control flow and artifact writers end to end, with
        # only expensive gradient/model/generation operations replaced offline.
        adj = sp.diags([np.ones(5), np.ones(5)], [-1, 1], shape=(6, 6)).tocsr()
        features = np.ones((6, 2), dtype=np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        train, evaluate = np.array([0]), np.arange(1, 6)
        digest = hashlib.sha256(b'offline integration').hexdigest()
        provenance = dict(protocol=PROTOCOL, sources={'heir.py': digest},
            input_files={'synthetic.npz': digest}, python_executable='/offline/python',
            python_environment={'python': 'fixture'}, api_base_url='http://offline.invalid',
            generation_model='offline', machine='fixture', cuda_visible_devices=None,
            thread_count=1, generation_temperature=0.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'results'
            for job in make_jobs(['citeseer'], [15], [.5]):
                run = job_directory(root, job)
                attack = self.make_attack(run/'artifacts', job['variant'])
                attack.args.dataset = 'citeseer'
                attack.options.trajectory = run.parent/'structure/artifacts/trajectory.json'
                attack.options.global_transition = 'attribute'
                attack.M = 6; attack.attack_structure = True
                attack.attack_features = True; attack.use_text_attack = True
                gradient = torch.zeros((6, 6))
                gradient[0, 5] = gradient[5, 0] = 5
                gradient[2, 3] = gradient[3, 2] = -4
                attack.get_meta_grad = lambda *a, **kw: (gradient, None)
                def writer(**kw):
                    attack.finish_text_attack_round(dict(targets=4, feature_writes=4,
                        feature_changes=0, template_attempts=1, template_successes=1,
                        template_failures=0, node_failures=0, cluster_failures=0,
                        cache_hits=0, completed=0))
                    return kw['full_features']
                attack.attack_features_with_text = writer
                with patch.object(Heirattack, 'inner_train', return_value=None), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if job['variant'] == 'structure':
                        with self.assertRaises(StructureReady):
                            attack.meta_attack_multi_step(features, adj, labels, train, evaluate, 2)
                        self.assertEqual(attack.actions, [[0, 5, 1], [2, 3, 0]])
                    else:
                        attack.meta_attack_multi_step(features, adj, labels, train, evaluate, 2)
                        logits = np.log(np.where(np.arange(2)[None, :] == labels[:, None], .9, .1))
                        for name in ('Clean graph', 'Edge-only attack', 'Feature-only attack', 'Combined attack'):
                            attack.record_victim_evaluation(name, logits, labels, evaluate)
                stage = load_stage(run, job)
                trajectory_hash = None if job['variant'] == 'structure' else stage['attack']['trajectory_hash']
                record = dict(**job, protocol=PROTOCOL, settings=protocol_settings(),
                    status='complete', returncode=0, provenance=provenance,
                    command=['offline integration'], wall_time_seconds=.1, peak_rss_mib=1., **stage)
                record['fingerprint'] = fingerprint(dict(job=job, settings=protocol_settings(),
                    provenance=provenance, trajectory_hash=trajectory_hash))
                (run/'result.json').write_text(json.dumps(record))
                self.assertEqual(validate_result(run/'result.json', job), record)
            output = Path(directory)/'reports'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(summarize(['--root', str(root), '--output-dir', str(output),
                    '--datasets', 'citeseer', '--seeds', '15', '--budgets', '.5']), 0)
            self.assertTrue((output/'paired_contrasts.csv').is_file())
            self.assertIn('preliminary', (output/'summary.md').read_text().lower())


if __name__=='__main__':unittest.main()
