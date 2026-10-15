"""Offline behavior checks for repetition-aware edit-gain selectors."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from gbhaa_experiments import selector_mechanisms as m
from gbhaa_experiments.post_tuning_protocol import get_profile, make_jobs, profile_catalog, runtime_settings, operator_settings

NEW = ('frontier_editgain_decay', 'frontier_editgain_fresh3')
ROOT = Path(__file__).resolve().parents[1]


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.states=np.array([[.3,.25,.1,.09,.08,.07,.05,.03,.02,.01]],np.float32)
        self.gains=np.array([0,0,30,20,15,8,7,6,10000,0],np.float64)
        self.visits={2:4,3:2,4:1}

    def order(self,rule,visits=None,gains=None,pool=6):
        self.assertIn(rule,m.RULES)
        return m.order_from_states(self.states,[0,1],self.visits if visits is None else visits,rule,pool,
                                   self.gains if gains is None else gains)

    def test_soft_decay_uses_prior_visit_count_with_same_local_pool(self):
        order,_,info=self.order(NEW[0])
        self.assertEqual(order[:4].tolist(),[5,4,6,3])
        self.assertEqual(info['candidate_pool'],[2,3,4,5,6,7])
        self.assertEqual(info['selected_prior_visits'],[0,1,0,2])
        np.testing.assert_allclose(info['selected_adjusted_gains'],[8,7.5,7,20/3])
        self.assertEqual(info['fresh_pool_count'],3)
        self.assertEqual(info['selected_fresh_count'],2)
        self.assertEqual(sorted(order.tolist()),list(range(10)))

    def test_fresh_quota_changes_selection_despite_large_repeated_gains(self):
        order,_,info=self.order(NEW[1])
        self.assertEqual(order[:4].tolist(),[5,6,7,2])
        self.assertEqual(info['fresh_quota_requested'],3)
        self.assertEqual(info['fresh_quota_effective'],3)
        self.assertEqual(info['fresh_quota_shortfall'],0)
        self.assertEqual(info['selected_fresh_count'],3)
        self.assertNotIn(8,order[:4]) # huge gain outside fixed pool cannot enter

    def test_all_fresh_first_round_matches_original_gain_ranking(self):
        expected=m.order_from_states(self.states,[0,1],{},'frontier_editgain',6,self.gains)[0]
        for rule in NEW:
            order,_,info=self.order(rule,visits={})
            np.testing.assert_array_equal(order,expected)
            self.assertEqual(info['selected_fresh_count'],4)

    def test_zero_gains_keep_quota_but_soft_decay_falls_back_to_diffusion(self):
        order,_,info=self.order(NEW[0],gains=np.zeros(10))
        self.assertEqual(order[:4].tolist(),[2,3,4,5])
        self.assertTrue(info['zero_gain_fallback'])
        order,_,info=self.order(NEW[1],gains=np.zeros(10))
        self.assertEqual(order[:4].tolist(),[5,6,7,2])
        self.assertTrue(info['zero_gain_fallback'])

    def test_scarce_fresh_pool_and_no_fresh_pool_do_not_break_four_writes(self):
        visits={u:1 for u in range(2,8)};visits[7]=0
        order,_,info=self.order(NEW[1],visits=visits)
        self.assertEqual(order[:4].tolist(),[7,2,3,4])
        self.assertEqual(info['fresh_quota_effective'],1)
        self.assertEqual(info['fresh_quota_shortfall'],2)
        order,_,info=self.order(NEW[1],visits={u:1 for u in range(2,8)})
        self.assertEqual(order[:4].tolist(),[2,3,4,5])
        self.assertEqual(info['fresh_quota_shortfall'],3)

    def test_max_visits_excluded_and_low_mass_fallback_preserved(self):
        for rule in NEW:
            order,_,info=self.order(rule,visits={2:10})
            self.assertNotIn(2,order[:4]);self.assertNotIn(2,info['candidate_pool'])
            states=np.array([[.6,.3,.1,0,0,0]],np.float32)
            order,_,info=m.order_from_states(states,[0,1],{},rule,32,np.ones(6))
            self.assertEqual(order[:4].tolist(),[2,0,1,3])
            self.assertEqual(info['fallback_count'],3)
            self.assertEqual(sorted(order.tolist()),list(range(6)))

    def test_multiround_quota_improves_coverage_in_repeated_high_gain_fixture(self):
        n=30;states=np.array([np.linspace(1,.1,n)],np.float32);gains=np.linspace(100,1,n)
        totals={}
        for rule in ('frontier_editgain',*NEW):
            self.assertIn(rule,m.RULES)
            visits=Counter()
            for _ in range(6):
                order,_,info=m.order_from_states(states,[0,1],visits,rule,28,gains)
                self.assertEqual(len(set(order[:4])),4)
                if rule==NEW[1]:self.assertGreaterEqual(info['selected_fresh_count'],3)
                visits.update(order[:4].tolist())
            self.assertEqual(sum(visits.values()),24);self.assertLessEqual(max(visits.values()),10)
            totals[rule]=len(visits)
        self.assertGreater(totals[NEW[0]],totals['frontier_editgain'])
        self.assertGreaterEqual(totals[NEW[1]],18)

    def test_profiles_preserve_model_budget_and_old_preset(self):
        self.assertTrue(set(NEW).issubset(profile_catalog()))
        old=get_profile('frontier_editgain')
        for name in NEW:
            config=get_profile(name)
            self.assertEqual(runtime_settings(config),runtime_settings(old))
            for k in old:
                if k not in ('id','description','selector_rule'):self.assertEqual(config[k],old[k])
            self.assertEqual(operator_settings(config)['selector_version'],'endpoint-frontier-coverage-v1')
        self.assertEqual(operator_settings(old)['selector_version'],'endpoint-frontier-v1')
        jobs=make_jobs(['citeseer'],[15,16],.1,60,preset='coverage',matched_controls=True)
        self.assertEqual(len(jobs),24)
        self.assertEqual({j['profile_id'] for j in jobs},{'baseline','frontier_editgain',*NEW})
        self.assertEqual(len(make_jobs(['citeseer'],[15,16],.1,60,preset='mechanism',matched_controls=True)),24)

    def test_launcher_preview_and_summary_route(self):
        script=ROOT/'gbhaa_experiments/launchers/run_post_coverage.sh'
        self.assertTrue(script.exists())
        for seeds,count in (([],24),(['--seeds','15'],12)):
            with tempfile.TemporaryDirectory() as temp:
                result_root=Path(temp)/'uncreated'
                p=subprocess.run(['bash',str(script),*seeds,'--output-root',str(result_root)],cwd=ROOT,
                    env={**os.environ,'PYTHON_BIN':sys.executable},capture_output=True,text=True,check=True)
                self.assertIn(f'PLAN ONLY: {count} short trials, 60 actions each',p.stdout)
                self.assertEqual(p.stdout.count('-m gbhaa_experiments.entrypoints.meta_post_tuning'),count)
                self.assertIn('--tune-profile frontier_editgain_fresh3 --tune-arm parallel',p.stdout)
                self.assertFalse(result_root.exists())
        p=subprocess.run(['bash',str(script),'summary','--help'],cwd=ROOT,
            env={**os.environ,'PYTHON_BIN':'/bin/echo'},capture_output=True,text=True,check=True)
        self.assertIn('post_coverage_results',p.stdout);self.assertIn('followup_reports/post_coverage',p.stdout)


class CoverageIntegrationTests(unittest.TestCase):
    def test_summary_compares_new_post_to_original_editgain_post(self):
        from unittest.mock import patch
        from test_post_tuning_summary import PostTuningSummaryTests
        from gbhaa_experiments import summarize_post_tuning as summary
        fixture=PostTuningSummaryTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        rows=fixture.fixture(profiles=('baseline','frontier_editgain',*NEW),matched=True)
        def read_validated(path,job):return json.loads(path.read_text())
        # Isolate aggregation after validation; real artifacts and loader are
        # exercised independently by replay/trace integration tests.
        with patch.object(summary,'validate_result',side_effect=read_validated), \
             patch.object(summary,'_per_trial',side_effect=lambda job,*a,**kw:job):
            report=summary.summarize(fixture.root)
        self.assertEqual(report['status'],'complete')
        for rule in NEW:
            comparisons=[c for c in report['comparisons'] if c['profile_id']==rule]
            self.assertEqual({(c['control_profile_id'],c['control_arm']) for c in comparisons},
                {('baseline','endpoint_post'),('frontier_editgain','endpoint_post'),(rule,'endpoint_pre'),(rule,'parallel')})

    def test_three_arm_visit_history_reconstruction_and_tamper_rejection(self):
        import scipy.sparse as sp
        from gbhaa_experiments.action_state_protocol import flip_action
        from gbhaa_experiments.selector_mechanism_validation import audit_mechanism_rounds
        from test_selector_mechanism_integration import MechanismIntegrationTests
        fixture=MechanismIntegrationTests()
        for rule in NEW:
            for arm in ('endpoint_post','endpoint_pre','parallel'):
                with self.subTest(rule=rule,arm=arm),tempfile.TemporaryDirectory() as path:
                    attack,x,info=fixture.setup_attack(rule,arm,path)
                    self.assertIsNotNone(info,'New guided rule must prepare frozen surrogate metadata')
                    attack.diagnostics=[]
                    original=sp.csr_matrix(np.ones((9,9),np.float32)-np.eye(9,dtype=np.float32))
                    actions=[[0,1,0],[2,3,0],[4,5,0]];state=original;updates=[]
                    for i,action in enumerate(actions):
                        attack.round_index=i;attack.diagnostics.append(dict(round=i,action=action))
                        post=flip_action(state,action)
                        nodes=attack._select_round(x,state,post,action)
                        for u in nodes:attack._attacked_nodes[int(u)]=attack._attacked_nodes.get(int(u),0)+1
                        updates.append(nodes);state=post
                    metadata=dict(config=attack.config,variant=arm,seed=15,actions=actions,selector_surrogate=info)
                    nodes=np.stack(updates);features=(x.numpy(),x.numpy(),nodes,np.stack([x.numpy()[v] for v in updates]))
                    path=Path(path)
                    (path/'rounds.json').write_text(json.dumps({'rounds':attack.diagnostics}))
                    audit_mechanism_rounds(path,metadata,original,features,attack.splits)
                    key={'endpoint_post':'post','endpoint_pre':'pre','parallel':'parallel'}[arm]
                    selections=[r['mechanism_traces'][key]['selection'] for r in attack.diagnostics]
                    self.assertTrue(any(sum(s['selected_prior_visits'])>0 for s in selections[1:]))
                    for field in ('selected_prior_visits','fresh_pool_count'):
                        corrupt=copy.deepcopy(attack.diagnostics)
                        selection=corrupt[1]['mechanism_traces'][key]['selection']
                        if field=='selected_prior_visits':selection[field][0]+=1
                        else:selection[field]+=1
                        (path/'rounds.json').write_text(json.dumps({'rounds':corrupt}))
                        with self.assertRaises(ValueError):
                            audit_mechanism_rounds(path,metadata,original,features,attack.splits)


if __name__=='__main__':unittest.main()
