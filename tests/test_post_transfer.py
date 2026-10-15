"""Small synthetic checks; no network and no real dataset training."""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

from gbhaa_experiments.projected_editing import adapt_candidate
from gbhaa_experiments.post_tuning_protocol import get_profile, make_jobs
from gbhaa_experiments.projected_editing_runtime import prepare_editor
from gbhaa_experiments.projected_editing_validation import audit_editor
import test_projected_editing_integration as integration
import test_post_editing_end_to_end as end_to_end

ROOT = Path(__file__).resolve().parents[1]
NEW = ('transfer_relu', 'transfer_residual', 'transfer_both')


class ResidualTests(unittest.TestCase):
    def test_only_planned_support_changes_even_after_projection(self):
        vocab = ['apple', 'berry', 'cherry', 'dates']
        vec = CountVectorizer(vocabulary=vocab, binary=True)
        current = np.array([1, 0, .023, 0, 1, .042, 1], np.float32)
        plan = dict(add=[1], drop=[0], keep=[], guided=True)
        out, info = adapt_candidate('cherry dates', plan, current, vec, vocab, 1, .97, residual=True)
        np.testing.assert_array_equal(out[[2,3,4,5,6]], current[[2,3,4,5,6]])
        self.assertGreater(out[1], 0); self.assertLess(out[0], 1)
        self.assertGreaterEqual(float(current@out/np.linalg.norm(current)/np.linalg.norm(out)), .969999)
        self.assertEqual(info['unplanned_decreased_entries'], 0)
        self.assertEqual(info['unplanned_increased_entries'], 0)
        self.assertEqual(info['unmapped_changed_entries'], 0)
        self.assertLess(info['projection_fraction'], 1)
        self.assertEqual(info['edit_policy'], 'planned_residual')

    def test_empty_plan_is_exact_noop_and_unsupported_policy_fails(self):
        vocab = ['apple', 'berry']; vec = CountVectorizer(vocabulary=vocab, binary=True)
        x = np.array([.23, 0, 1], np.float32)
        p = dict(add=[], drop=[], keep=[], guided=True)
        out, _ = adapt_candidate('apple berry', p, x, vec, vocab, 1, .99, residual=True)
        np.testing.assert_array_equal(out, x)
        with self.assertRaises(ValueError):
            adapt_candidate('apple', {**p, 'guided': False}, x, vec, vocab, 1, .35, residual=True)


class TransferTests(integration.EditingIntegrationTests):
    # Reuse real replay/audit fixtures, with train-only tiny proxy fits.
    def setup_attack(self, path, profile, arm):
        a, x, graph, info, snapshot, calls = super().setup_attack(path, 'edit_both', arm)
        a.config = get_profile(profile)
        a.args.hidden=a.surrogate.hidden_sizes[0];a.args.dropout=a.surrogate.dropout
        original_state = {k: v.detach().clone() for k,v in a.surrogate.state_dict().items()}
        rng = torch.random.get_rng_state().clone(); nrng = np.random.get_state(); prng = random.getstate()
        snapshot = prepare_editor(a, features=x, graph=graph, train_labels=np.array([0, 1]))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(prng, random.getstate()); np.testing.assert_array_equal(nrng[1], np.random.get_state()[1])
        for k,v in original_state.items(): self.assertTrue(torch.equal(v, a.surrogate.state_dict()[k]))
        self.assertEqual(info, a._selector_surrogate_info)
        return a, x, graph, info, snapshot, calls

    def test_proxy_fitter_receives_only_training_labels_and_no_validation(self):
        from deeprobust.graph.defense import GCN
        original=GCN.fit;seen=[]
        def capture(model,features,graph,labels,train,**kwargs):
            seen.append((model,labels.copy(),train.copy(),kwargs.copy()))
            return original(model,features,graph,labels,train,**kwargs)
        with tempfile.TemporaryDirectory() as path,patch.object(GCN,'fit',capture):
            a,x,g,*_=self.setup_attack(path,'transfer_relu','endpoint_pre')
            self.assertEqual(len(seen),1)
            model,labels,train,kwargs=seen[0]
            np.testing.assert_array_equal(labels[[0,1]],[0,1])
            np.testing.assert_array_equal(labels[2:],np.zeros(7,dtype=np.int64))
            np.testing.assert_array_equal(train,[0,1])
            self.assertIsNone(kwargs['idx_val']);self.assertEqual(kwargs['train_iters'],200)
            self.assertTrue(model.with_relu);self.assertEqual(model.dropout,a.surrogate.dropout)
            self.assertEqual(model.lr,.01);self.assertEqual(model.weight_decay,.0005)
            with self.assertRaises(ValueError):
                prepare_editor(a,features=x,graph=g,train_labels=np.arange(9)%2)

    def test_nondefault_victim_architecture_fails_before_fit(self):
        from deeprobust.graph.defense import GCN
        with tempfile.TemporaryDirectory() as path:
            a,x,g,*_=self.setup_attack(path,'transfer_relu','endpoint_post')
            a.args.hidden=7;a.args.dropout=.2
            with patch.object(GCN,'fit',side_effect=AssertionError('must reject before fitting')):
                with self.assertRaisesRegex(ValueError,'victim architecture'):
                    prepare_editor(a,features=x,graph=g,train_labels=np.array([0,1]))

    def test_cpu_proxy_does_not_seed_accelerators(self):
        with tempfile.TemporaryDirectory() as path:
            a,x,g,*_=self.setup_attack(path,'transfer_relu','endpoint_post')
            # torch.manual_seed seeds MPS/CUDA/XPU too; the CPU fitter must not use it.
            with patch('torch.manual_seed',side_effect=AssertionError('accelerator seed changed')):
                prepare_editor(a,features=x,graph=g,train_labels=np.array([0,1]))

    def test_new_dispatches_and_audits_and_source_is_unchanged(self):
        for profile in NEW:
            for arm in ('endpoint_post', 'endpoint_pre', 'parallel'):
                with self.subTest(profile=profile,arm=arm), tempfile.TemporaryDirectory() as path:
                    a,m,g,f,calls = self.fixture(path, profile, arm, 2)
                    self.assertEqual(len(calls), 6)
                    audit_editor(Path(path), m,g,f,a.splits)
                    self.assertEqual(a._editor_model.with_relu, profile != 'transfer_residual')
                    self.assertFalse(a.surrogate.with_relu)
                    if profile != 'transfer_relu':
                        for r in a.diagnostics:
                            for n in r['editing']['nodes']:
                                for p in n['projections']:
                                    self.assertEqual(p['unplanned_decreased_entries'],0)
                                    self.assertEqual(p['unplanned_increased_entries'],0)

    def test_relu_audit_rejects_receipt_and_source_checkpoint_tampering(self):
        with tempfile.TemporaryDirectory() as path:
            a,m,g,f,_ = self.fixture(path, 'transfer_both', 'endpoint_post')
            bad = copy.deepcopy(m); bad['editor_snapshot']['transfer']['seed'] += 1
            with self.assertRaises(ValueError): audit_editor(Path(path), bad,g,f,a.splits)
            file = Path(path)/'editor_source_weights.npz'
            with np.load(file) as z: weights={k:z[k] for k in z.files}
            weights['gc1.weight'][0,0] += .1; np.savez_compressed(file, **weights)
            with self.assertRaises(ValueError): audit_editor(Path(path), m,g,f,a.splits)


class TransferLauncherTests(unittest.TestCase):
    def test_matched_preview_and_explicit_new_output(self):
        jobs=make_jobs(['citeseer'], [15], .1, 60, preset='transfer', matched_controls=True)
        self.assertEqual(len(jobs),15)
        self.assertEqual({j['profile_id'] for j in jobs},{'edit_control','edit_both',*NEW})
        with tempfile.TemporaryDirectory() as p:
            out=Path(p)/'untouched'
            result=subprocess.run(['bash',str(ROOT/'gbhaa_experiments/launchers/run_post_transfer.sh'),
                '--output-root',str(out)],env={**os.environ,'PYTHON_BIN':sys.executable},
                capture_output=True,text=True,check=True)
            self.assertIn('PLAN ONLY: 15 short trials, 60 actions each',result.stdout)
            self.assertFalse(out.exists())
            for profile in NEW:
                for arm in ('endpoint_post','endpoint_pre','parallel'):
                    self.assertIn(f'--tune-profile {profile} --tune-arm {arm}',result.stdout)

    def test_transfer_factorial_reports_missing_seed_and_single_seed_honestly(self):
        from gbhaa_experiments.post_editing_summary import transfer_effects
        values={'edit_both':.8,'transfer_relu':.7,'transfer_residual':.6,'transfer_both':.4}
        data={(p,arm):{s:dict(evaluation=dict(metrics=dict(validation_accuracy=a+offset)))
             for s,offset in ((15,0),(16,.01))} for p,a in values.items()
             for arm in ('endpoint_post','endpoint_pre','parallel')}
        rows=transfer_effects(data,[15,16],('citeseer',.1,60),True)
        for r in rows:
            self.assertAlmostEqual(r['mean_pp'],{'proxy':15,'residual':25,'interaction':10}[r['effect']])
            self.assertIsNotNone(r['ci95_pp'])
        for r in transfer_effects(data,[15],('citeseer',.1,60),True):self.assertIsNone(r['ci95_pp'])
        del data['transfer_both','endpoint_post'][16]
        for r in transfer_effects(data,[15,16],('citeseer',.1,60),False):self.assertIsNone(r['mean_pp'])

    def test_public_thirty_trial_matrix_and_model_matching(self):
        helper=end_to_end.EditingEndToEndTests();self.addCleanup(helper.doCleanups)
        fixture,rows=helper.fixture(profiles=('edit_control','edit_both',*NEW))
        code,report,recommendation=fixture.summarize()
        self.assertEqual(code,0,report.get('rejected_blocks'))
        self.assertEqual(report['status'],'complete');self.assertEqual(report['expected_jobs'],30)
        self.assertEqual(len(report['transfer_effects']),9);self.assertFalse(report['factorial_effects'])
        self.assertTrue((fixture.output/'transfer_effects.csv').exists())
        for r in report['per_trial']:
            self.assertIn('editor_proxy_policy',r)
            if r['config'].get('editor_residual'):self.assertEqual(r['editor_unmapped_decreases_sum'],0)
        # Correct per-model grouping is required; changing one proxy's training
        # receipt alone must still reject the entire comparison block.
        row=next(r for r in rows if r['profile_id']=='transfer_both')
        row['attack']['editor_snapshot']['transfer']['seed']+=1;fixture.save(row)
        code,report,recommendation=fixture.summarize()
        self.assertEqual(code,2);self.assertFalse(recommendation['recommendations'])
        self.assertTrue(all(r['mean_pp'] is None for r in report['transfer_effects']))


if __name__=='__main__':unittest.main()
