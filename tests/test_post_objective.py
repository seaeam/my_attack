"""Objective pilot: analytic gradients and offline matched artifact replays."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from gbhaa_experiments.editor_objectives import objective_value, target_weights, objective_gradient
from gbhaa_experiments.post_tuning_protocol import (
    OBJECTIVE_PROFILES, get_profile, make_jobs, validate_config, reference_profile, diffusion_scores)
from gbhaa_experiments.projected_editing_runtime import edit_round
from gbhaa_experiments.projected_editing_validation import audit_editor
from gbhaa_experiments.action_state_protocol import independent_restarts, flip_action
from gbhaa_experiments.followup_protocol import array_hash
import test_post_transfer as transfer
import test_post_editing_end_to_end as end_to_end

ROOT = Path(__file__).resolve().parents[1]


class ObjectiveUnitTests(unittest.TestCase):
    def test_margin_plateaus_after_boundary_and_ignores_common_logit_offset(self):
        output = torch.tensor([[3., 1., 0.], [0., 2., 1.], [2., 2., 0.]], requires_grad=True)
        y = torch.tensor([0, 0, 0]); ids = torch.arange(3); w = torch.tensor([.2, .3, .5])
        score = objective_value(output, y, ids, w, 'capped_margin')
        self.assertAlmostEqual(float(score), -.4, places=6)
        gradient, = torch.autograd.grad(score, output)
        torch.testing.assert_close(gradient, torch.tensor([[-.2,.2,0.],[0.,0.,0.],[0.,0.,0.]]))
        self.assertAlmostEqual(float(objective_value(output + 10, y, ids, w, 'capped_margin')), -.4, places=6)

    def test_weighted_nll_and_feature_gradient_agree(self):
        class Model(torch.nn.Module):
            def forward(self, x, adj): return x.log_softmax(1)
        model = Model().train(); x = torch.tensor([[2.,0.],[1.,0.],[0.,3.]])
        ids = torch.tensor([1,2]); pseudo = torch.tensor([0,0,0]); w = torch.tensor([.25,.75])
        for kind in ('nll','capped_margin'):
            g, score = objective_gradient(model, x, None, pseudo, ids, w, kind)
            independent = x.clone().requires_grad_(True)
            out = model(independent, None)
            expected = (-.25*out[1,0]-.75*out[2,0] if kind=='nll' else
                        -.25*torch.relu(out[1,0]-out[1,1])-.75*torch.relu(out[2,0]-out[2,1]))
            expected_grad, = torch.autograd.grad(expected, independent)
            np.testing.assert_allclose(g, expected_grad.numpy(), atol=1e-7)
            self.assertAlmostEqual(score, float(expected), places=7)
            self.assertTrue(model.training); self.assertFalse(x.requires_grad)
            np.testing.assert_array_equal(g[0], 0.)

    def test_local_weights_restrict_then_normalize_and_fallback(self):
        x = np.ones((5,2), np.float32)
        g = sp.csr_matrix([[0,1,1,0,0],[1,0,0,0,0],[1,0,0,1,0],[0,0,1,0,0],[0,0,0,0,0]])
        targets = np.array([1,2,3,4]); config = get_profile('objective_local')
        weights, info = target_weights(x,g,[0,1],targets,config)
        mass = diffusion_scores(x,g,[0,1],config)[targets].astype(np.float64)
        np.testing.assert_allclose(weights, .5/4+.5*mass/mass.sum(), rtol=1e-6)
        self.assertEqual(info['target_nodes'], targets.tolist()); self.assertFalse(info['uniform_fallback'])
        self.assertEqual(info['weights_hash'], array_hash(weights))
        self.assertTrue(np.all(weights>0)); self.assertAlmostEqual(float(weights.sum()),1.)
        pre = target_weights(x,g,[0,1],targets,config)[0]
        post = target_weights(x,flip_action(g,[0,1,0]),[0,1],targets,config)[0]
        parallel = target_weights(x,g,[3,4],targets,config)[0]
        self.assertFalse(np.allclose(pre,post)); self.assertFalse(np.allclose(pre,parallel))
        uniform, fallback = target_weights(x,sp.csr_matrix((5,5)),[0],[1,2],config)
        np.testing.assert_array_equal(uniform,[.5,.5]); self.assertTrue(fallback['uniform_fallback'])
        with self.assertRaises(ValueError): target_weights(x,g,[],targets,config)
        with self.assertRaises(ValueError): target_weights(x,g,[0],[1,1],config)
        with self.assertRaises(ValueError): target_weights(x,g,[0],[1.1,2.1],config)

    def test_objective_requires_valid_targets_and_positive_normalized_weights(self):
        out=torch.zeros((3,2)); y=torch.zeros(3,dtype=torch.long)
        for ids,w,kind in (([0,0],[.5,.5],'nll'),([0,1],[.5,-.5],'nll'),
                ([0,1],[.4,.4],'nll'),([0,1],[0.,1.],'nll'),([0,1],[.5,.5],'unknown')):
            with self.subTest(ids=ids,w=w,kind=kind), self.assertRaises(ValueError):
                objective_value(out,y,torch.tensor(ids),torch.tensor(w),kind)

    def test_catalog_and_schema_and_legacy_reference(self):
        jobs = make_jobs(['citeseer'],[15,16],.1,60,preset='objective',matched_controls=True)
        self.assertEqual(len(jobs),24)
        self.assertEqual({j['profile_id'] for j in jobs},set(OBJECTIVE_PROFILES))
        self.assertEqual(reference_profile(OBJECTIVE_PROFILES),'transfer_relu')
        self.assertEqual(reference_profile(['transfer_relu']),'edit_control')
        c=get_profile('objective_both')
        for changes in ({'editor_local_mix':1.},{'editor_local_mix':float('nan')},
                        {'editor_objective':'fake'},{'editor_proxy':'source'},{'editor_residual':True}):
            with self.subTest(changes=changes),self.assertRaises(ValueError): validate_config({**c,**changes})


class ObjectiveReplayTests(unittest.TestCase):
    def test_all_arms_same_checkpoint_and_audit_rejects_wrong_restarts_or_weights(self):
        helper=transfer.TransferTests(); snapshots=[]
        for profile in OBJECTIVE_PROFILES[1:]:
            for arm in ('endpoint_post','endpoint_pre','parallel'):
                with self.subTest(profile=profile,arm=arm),tempfile.TemporaryDirectory() as path:
                    a,m,g,f,calls=helper.fixture(path,profile,arm,2)
                    snapshots.append(m['editor_snapshot']); self.assertEqual(len(calls),6)
                    self.assertEqual(a.attribute_counts['feature_writes'],8)
                    audit_editor(Path(path),m,g,f,a.splits)
                    info=a.diagnostics[0]['editing']['objective']
                    expected=independent_restarts(9,15,0) if arm=='parallel' else [0,1]
                    self.assertEqual(info['restart_nodes'],list(expected))
                    for key in ('restart_nodes','weights','graph_hash'):
                        d=copy.deepcopy(a.diagnostics)
                        if key=='graph_hash': d[0]['editing'][key]='0'*64
                        else: d[0]['editing']['objective'][key][0]+=1
                        (Path(path)/'rounds.json').write_text(json.dumps({'rounds':d}))
                        with self.assertRaises(ValueError): audit_editor(Path(path),m,g,f,a.splits)
        self.assertTrue(all(s==snapshots[0] for s in snapshots))

    def test_weights_frozen_once_and_candidate_scores_match_recorded_objective(self):
        from deeprobust.graph import utils
        import gbhaa_experiments.editor_objectives as objectives
        helper=transfer.TransferTests()
        with tempfile.TemporaryDirectory() as path:
            a,x,g,*_=helper.setup_attack(path,'objective_both','endpoint_post')
            pseudo=torch.zeros(9,dtype=torch.long); ids=torch.arange(2,9)
            with patch.object(objectives,'target_weights',wraps=target_weights) as compute:
                result,trace,bank=edit_round(x,g,[0,1,2,3],a._editor_model,pseudo,ids,
                    a.text_generator.vectorizer,a.text_generator.vocab,a.config,15,0,
                    lambda _: 'apple berry',restarts=[0,1])
            self.assertEqual(compute.call_count,1)
            weights=torch.tensor(trace['objective']['weights']); current=x.clone()
            normalized=utils.normalize_adj_tensor(utils.sparse_mx_to_torch_sparse_tensor(g),sparse=True)
            for i,node in enumerate(trace['nodes']):
                scores=[]
                for candidate in bank[i]:
                    trial=current.clone(); trial[node['node']]=torch.from_numpy(candidate)
                    with torch.no_grad():
                        scores.append(float(objective_value(a._editor_model(trial,normalized),pseudo,ids,weights,'capped_margin')))
                np.testing.assert_array_equal(scores,node['candidate_losses'])
                self.assertEqual(node['chosen'],int(np.argmax(scores)))
                current[node['node']]=torch.from_numpy(bank[i,node['chosen']])
            torch.testing.assert_close(result,current)


class ObjectiveReportTests(unittest.TestCase):
    def test_joint_wins_require_both_strictly_better_and_incomplete_suppresses(self):
        from gbhaa_experiments.post_editing_summary import joint_advantage
        def rows(values): return {s:dict(evaluation=dict(metrics=dict(validation_accuracy=a))) for s,a in zip([15,16],values)}
        data={('x','endpoint_post'):rows([.6,.7]),('x','endpoint_pre'):rows([.7,.8]),('x','parallel'):rows([.65,.7])}
        row=joint_advantage(data,[15,16],('citeseer',.1,60),True,['x'])[0]
        self.assertEqual(row['strict_joint_wins'],1)
        self.assertAlmostEqual(row['minimum_advantage_mean_pp'],2.5)
        self.assertEqual(row['worst_seed_minimum_advantage_pp'],0.)
        self.assertIsNotNone(row['minimum_advantage_ci95_pp'])
        row=joint_advantage(data,[15,16],('citeseer',.1,60),False,['x'])[0]
        self.assertIsNone(row['minimum_advantage_mean_pp']); self.assertIsNone(row['strict_joint_wins'])

    def test_public_24_trial_matrix_and_missing_result(self):
        helper=end_to_end.EditingEndToEndTests();self.addCleanup(helper.doCleanups)
        fixture,rows=helper.fixture(profiles=OBJECTIVE_PROFILES)
        code,report,recommendation=fixture.summarize()
        self.assertEqual(code,0,report.get('rejected_blocks'))
        self.assertEqual(report['expected_jobs'],24);self.assertEqual(len(report['objective_effects']),9)
        self.assertFalse(report['transfer_effects']);self.assertFalse(report['factorial_effects'])
        self.assertEqual(len(report['joint_advantage']),4)
        self.assertTrue((fixture.output/'objective_effects.csv').exists())
        self.assertTrue((fixture.output/'joint_advantage.csv').exists())
        (fixture.directory(rows[0])/'result.json').unlink()
        code,report,recommendation=fixture.summarize()
        self.assertEqual(code,2);self.assertFalse(recommendation['recommendations'])
        self.assertTrue(all(r['mean_pp'] is None for r in report['objective_effects']))
        self.assertTrue(all(r['strict_joint_wins'] is None for r in report['joint_advantage']))

    def test_launcher_preview_and_one_seed_override(self):
        with tempfile.TemporaryDirectory() as path:
            out=Path(path)/'untouched'
            for extra,count in (([],24),(['--seeds','15'],12)):
                result=subprocess.run(['bash',str(ROOT/'gbhaa_experiments/launchers/run_post_objective.sh'),
                    '--output-root',str(out),*extra],env={**os.environ,'PYTHON_BIN':sys.executable},
                    capture_output=True,text=True,check=True)
                self.assertIn(f'PLAN ONLY: {count} short trials, 60 actions each',result.stdout)
                self.assertFalse(out.exists())


if __name__=='__main__': unittest.main()
