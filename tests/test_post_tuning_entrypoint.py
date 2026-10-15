"""Offline coverage of tuning replay and validation-only evaluation."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from heir import Heirattack
from gbhaa_experiments.entrypoints.meta_post_tuning import (
    TuningAttack, validate_splits, evaluate_validation,
)
from gbhaa_experiments.post_tuning_protocol import get_profile, runtime_settings
from gbhaa_experiments.entrypoints.meta_action_state import ActionStateAttack
from gbhaa_experiments.entrypoints.meta_followup import FollowupAttack
import test_action_state_summary as original_fixtures


class TuningEntrypointTests(unittest.TestCase):
    def test_generation_profile_reaches_live_generator_and_request_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            def initialize(attack):
                torch.nn.Module.__init__(attack)
                attack.options=SimpleNamespace(profile_id='tokens150',cache=Path(directory))
                generator=SimpleNamespace(max_tokens=50,model_name='fixture',generation_seed=1,generation_temperature=0.)
                generator.generate_text=lambda messages:f'template with limit {generator.max_tokens}'
                attack.text_generator=generator
            with patch.object(FollowupAttack,'__init__',initialize):
                attack=TuningAttack()
            self.assertEqual(attack.text_generator.max_tokens,150)
            self.assertEqual(attack.text_generator.generate_text([{'role':'user','content':'fixture'}]),'template with limit 150')
            records=list(Path(directory).glob('*.json'))
            self.assertEqual(len(records),1)
            self.assertEqual(json.loads(records[0].read_text())['request']['max_tokens'],150)

    def test_baseline_selection_matches_original_all_arms_with_visit_filter(self):
        rng=np.random.default_rng(22)
        features=torch.tensor(rng.uniform(size=(8,4)),dtype=torch.float32)
        pre=sp.csr_matrix(np.ones((8,8))-np.eye(8))
        post=pre.tolil();post[0,1]=post[1,0]=0;post=post.tocsr()
        for arm in ('parallel','endpoint_pre','endpoint_post'):
            attack=TuningAttack.__new__(TuningAttack);torch.nn.Module.__init__(attack)
            attack.config=get_profile('baseline');attack.args=SimpleNamespace(seed=15)
            attack.options=SimpleNamespace(phase=arm);attack.device='cpu';attack.nnodes=8
            attack.round_index=3;attack._attacked_nodes={0:10,2:3}
            attack.text_attack_max_visits=10;attack.text_attack_total_nodes=None
            attack.diagnostics=[{}]
            expected=ActionStateAttack._select_round(attack,features,pre,post,[0,1,0])
            actual=TuningAttack._select_round(attack,features,pre,post,[0,1,0])
            np.testing.assert_array_equal(actual,expected)

    def test_splits_reject_overlap_empty_duplicate_and_invalid_nodes(self):
        split = validate_splits(6, [0, 1], [2, 3], [4, 5])
        np.testing.assert_array_equal(split['validation'], [2, 3])
        for train, val, test in [([0, 1], [1, 2], [3, 4, 5]),
                                 ([0, 1], [], [2, 3, 4, 5]),
                                 ([0, 0], [2, 3], [4, 5]),
                                 ([0, 1], [2, 3], [4, 6]),
                                 ([0., 1.], [2, 3], [4, 5])]:
            with self.assertRaises(ValueError):
                validate_splits(6, train, val, test)

    def test_partial_splits_preserve_ids_without_assigning_remaining_graph_nodes(self):
        # The cached Citeseer nettack split has 210 + 211 + 1688 IDs for 2110 nodes.
        train=np.arange(210,dtype=np.int64)[::-1]
        validation=np.arange(210,421,dtype=np.int64)
        test=np.arange(421,2109,dtype=np.int64)
        splits=validate_splits(2110,train,validation,test)
        for name,expected in (('train',train),('validation',validation),('test',test)):
            np.testing.assert_array_equal(splits[name],expected)
        self.assertNotIn(2109,np.concatenate(list(splits.values())))

    def test_validation_metrics_ignore_test_and_unassigned_labels(self):
        class Victim:
            def __init__(self, **kwargs): pass
            def to(self, device): return self
            def fit(self, features, adj, labels, idx_train):
                np.testing.assert_array_equal(labels,[0,1,0,0,0,0,0])
                self.output = torch.log(torch.tensor([[.8, .2], [.2, .8], [.7, .3], [.6, .4], [.1, .9], [.9, .1], [.5,.5]]))
        with tempfile.TemporaryDirectory() as directory:
            attack = SimpleNamespace(trace_dir=Path(directory), args=SimpleNamespace(seed=15,hidden=16,dropout=.5),
                device='cpu', splits=validate_splits(7,[0,1],[2,3],[4,5]))
            original = np.ones((7, 2), np.float32); adj = sp.eye(7, format='csr')
            labels = torch.tensor([0,1,0,1,0,1,1])
            first = evaluate_validation(attack, original, adj, original, adj, labels, victim_class=Victim)
            changed = labels.clone(); changed[4:] = 1-changed[4:]
            second = evaluate_validation(attack, original, adj, original, adj, changed, victim_class=Victim)
            self.assertEqual(first, second)
            self.assertEqual(first['metrics']['validation_accuracy'], .5)
            with np.load(Path(directory)/'validation_predictions.npz') as z:
                np.testing.assert_array_equal(z['node_ids'], [2,3])
            self.assertFalse(list(Path(directory).glob('*test*')))

    def test_real_replay_reuses_prefix_and_records_actual_profile(self):
        fixtures = original_fixtures.ActionStateSummaryTests()
        fixtures.setUp(); self.addCleanup(fixtures.doCleanups)
        parent = fixtures.fixture()[0]
        source_run = fixtures.directory(parent)
        with np.load(source_run.parent/'parallel/artifacts/features.npz') as f:
            features = f['original']
        adj = sp.load_npz(source_run/'artifacts/original_adj.npz')
        final_hashes = []
        for profile, arm in itertools.product(('floor025','frontier','balanced_frontier','frontier_editgain',
                                                'frontier_editgain_decay','frontier_editgain_fresh3'),
                                               ('parallel','endpoint_pre','endpoint_post')):
            with tempfile.TemporaryDirectory() as directory:
                cfg = get_profile(profile)
                attack = TuningAttack.__new__(TuningAttack); torch.nn.Module.__init__(attack)
                attack.config = cfg
                attack.args = SimpleNamespace(dataset='citeseer', seed=15, ptb_rate=.05, **runtime_settings(cfg))
                attack.options = SimpleNamespace(phase=arm, artifacts=Path(directory), source_run=source_run,
                    max_actions=1, profile_id=profile, trial_id=f'{profile}__{arm}')
                attack.trace_dir=Path(directory); attack.nnodes=12; attack.device='cpu'
                attack.freeze_structure_features=True; attack.use_oracle=False; attack.text_attack_total_nodes=None
                attack.text_generator=SimpleNamespace(model_name='fixture',max_tokens=50,api_stats={},
                    uses_fallback_vocabulary=False, uses_partial_vocabulary=False, vocab=list(range(features.shape[1])))
                class FixedSurrogate(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.weight=torch.nn.Parameter(torch.arange(features.shape[1]*2,dtype=torch.float32).reshape(-1,2)/10)
                        self.output=self(torch.tensor(features),torch.eye(len(features))).detach()
                    def forward(self,x,a):return torch.log_softmax(a@x@self.weight,dim=1)
                attack.surrogate=FixedSurrogate()
                attack.request_cache=SimpleNamespace(stats={'hits':0,'misses':0},used_keys=set())
                attack.attribute_counts={}; attack.actions=[];attack.diagnostics=[];attack.evaluations={}
                attack.round_index=-1;attack.template_cache={}
                for key,value in runtime_settings(cfg).items():setattr(attack,key,value)
                attack.self_training_label=lambda y,idx:y
                attack.get_embeddings=lambda x,a:x
                attack._build_round_text_clusters=lambda nodes,*a,**kw:{0:nodes.tolist()}
                attack.configure_evaluation_split(np.arange(6,12),np.arange(3),np.arange(3,6))
                def writer(**kw):
                    attack.finish_text_attack_round(dict(targets=4,feature_writes=4,feature_changes=0,
                        template_attempts=1,template_successes=1,template_failures=0,node_failures=0,
                        cluster_failures=0,cache_hits=0,completed=0))
                    return kw['full_features']
                attack.attack_features_with_text=writer
                with patch.object(Heirattack,'inner_train',return_value=None), contextlib.redirect_stdout(io.StringIO()):
                    attack.meta_attack_multi_step(features,adj,np.zeros(12,dtype=np.int64),
                        np.arange(6,12),np.arange(6),1)
                metadata=json.loads((Path(directory)/'attack.json').read_text())
                self.assertEqual(metadata['actions'],parent['trajectory']['actions'][:1])
                self.assertEqual(metadata['config'],cfg)
                self.assertEqual(metadata['target_edges'],1)
                self.assertAlmostEqual(metadata['effective_budget'],1/40)
                self.assertEqual(metadata['attribute_node_visits'],4)
                self.assertEqual(metadata['source_budget'],.05)
                self.assertEqual(metadata['validation_nodes_hash'],attack.metadata['validation_nodes_hash'])
                self.assertEqual(metadata['operator_settings']['weight_floor'],cfg['weight_floor'])
                if 'selector_rule' in cfg:
                    from gbhaa_experiments.selector_mechanism_validation import audit_mechanism_rounds
                    with np.load(Path(directory)/'feature_updates.npz') as updates:
                        audit_mechanism_rounds(Path(directory),metadata,adj,
                            (features,features,updates['nodes'],updates['values']),attack.splits)
                class FixedVictim:
                    def __init__(self,**kw):pass
                    def to(self,device):return self
                    def fit(self,x,a,y,idx):
                        self.output=torch.log(torch.tensor([[.7,.3]]*len(x)))
                attack.args.hidden=16;attack.args.dropout=.5
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluate_validation(attack,features,adj,features,attack.modified_adj,
                                        np.zeros(12,dtype=np.int64),victim_class=FixedVictim)
                # Full loader checks parent identity, replay, counts, splits,
                # new selector traces and predictions together.
                from gbhaa_experiments.post_tuning_validation import load_trial
                from gbhaa_experiments.post_tuning_protocol import make_jobs
                with tempfile.TemporaryDirectory() as copied:
                    shutil.copytree(directory,Path(copied)/'artifacts')
                    job=next(j for j in make_jobs(['citeseer'],[15],.05,1,profile_ids=[profile],matched_controls=True)
                             if j['profile_id']==profile and j['arm']==arm)
                    checked=load_trial(Path(copied),job,parent_run=source_run)
                    self.assertEqual(checked['attack'],metadata)
                    from gbhaa_experiments.summarize_post_tuning import _per_trial
                    exported=_per_trial(job,'complete',Path(copied)/'result.json',
                        {**checked,'wall_time_seconds':0.,'peak_rss_mib':None})
                    self.assertEqual(exported['fresh_node_write_fraction'],1.)
                    self.assertEqual(exported['repeated_node_write_fraction'],0.)
                    if profile=='frontier_editgain_fresh3':
                        key={'parallel':'parallel','endpoint_pre':'pre','endpoint_post':'post'}[arm]
                        selection=attack.diagnostics[0]['mechanism_traces'][key]['selection']
                        # An empty local pool can have a quota shortfall even
                        # though legal fallback supplies four first visits.
                        self.assertEqual(exported['mean_fresh_quota_shortfall'],selection['fresh_quota_shortfall'])

                final_hashes.append(metadata['modified_adj_hash'])
        self.assertEqual(len(set(final_hashes)),1)


if __name__=='__main__':unittest.main()
