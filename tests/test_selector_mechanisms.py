"""Offline tests for explicit selector method variants; no fitting or generation."""
import importlib
import importlib.util
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from gbhaa_experiments.post_tuning_protocol import get_profile, make_jobs, validate_config


class SelectorMechanismTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('gbhaa_experiments.selector_mechanisms'))
        return importlib.import_module('gbhaa_experiments.selector_mechanisms')

    def test_fixed_budget_matched_two_seed_grid_and_legacy_identity(self):
        from gbhaa_experiments.post_tuning_protocol import profile_catalog, build_trials
        self.assertIn('frontier', profile_catalog())
        jobs=make_jobs(['citeseer'],[15,16],.1,60,preset='mechanism',matched_controls=True)
        self.assertEqual(len(jobs),24)
        self.assertEqual({j['profile_id'] for j in jobs},
                         {'baseline','frontier','balanced_frontier','frontier_editgain'})
        base=get_profile('baseline')
        for name in ('frontier','balanced_frontier','frontier_editgain'):
            cfg=get_profile(name)
            self.assertEqual(cfg['selector_rule'],name)
            self.assertEqual(cfg['selector_pool_size'],32)
            self.assertTrue(all(cfg[k]==base[k] for k in base if k not in ('id','description','family')))
        self.assertNotIn('selector_rule',base)
        self.assertEqual([len(build_trials(p)) for p in ('tiny','quick','extended','refine')],[6,12,22,12])

    def test_frontier_uses_positive_nonseed_support_then_reachable_fallback(self):
        m=self.module()
        states=np.array([[.4,.35,.1,.08,.04,.02,.01,0]],np.float32)
        order,score,info=m.order_from_states(states,[0,1],{},'frontier',32)
        np.testing.assert_array_equal(order[:4],[2,3,4,5])
        self.assertEqual(info['fallback_count'],0)
        small=np.array([[.6,.3,.1,0,0,0]],np.float32)
        order,_,info=m.order_from_states(small,[0,1],{},'frontier',32)
        np.testing.assert_array_equal(order[:4],[2,0,1,3])
        self.assertEqual(info['fallback_count'],3)
        self.assertEqual(sorted(order.tolist()),list(range(6)))

    def test_visits_filter_before_candidate_pool_and_all_zero_gain_falls_back(self):
        m=self.module();states=np.array([[.3,.25,.2,.1,.07,.04,.03,.01]],np.float32)
        gain=np.arange(8,dtype=np.float32)
        order,_,info=m.order_from_states(states,[0,1],{2:10},'frontier_editgain',4,gain)
        self.assertEqual(info['candidate_pool'],[3,4,5,6])
        self.assertEqual(order[:4].tolist(),[6,5,4,3])
        self.assertNotIn(2,order[:4])
        order,_,info=m.order_from_states(states,[0,1],{2:10},'frontier_editgain',4,np.zeros(8))
        self.assertEqual(order[:4].tolist(),[3,4,5,6])
        self.assertTrue(info['zero_gain_fallback'])

    def test_balanced_frontier_interleaves_branches_and_is_restart_swap_invariant(self):
        m=self.module()
        states=np.array([[.35,.01,.3,.2,.1,.02,.01,.01],
                         [.01,.35,.01,.01,.02,.3,.2,.1]],np.float32)
        order,_,_=m.order_from_states(states,[0,1],{},'balanced_frontier',32)
        self.assertEqual(order[:4].tolist(),[2,5,3,6])
        swapped,_,_=m.order_from_states(states[::-1],[1,0],{},'balanced_frontier',32)
        np.testing.assert_array_equal(order,swapped)

    def test_order_rejects_bad_states_gains_restarts_and_exhausted_budget(self):
        m=self.module();states=np.array([[.3,.3,.1,.1,.1,.1]],np.float32)
        invalid=[(states,[0,0],{},'frontier',32,None),
                 (states,[0,1],{},'missing',32,None),
                 (states*float('nan'),[0,1],{},'frontier',32,None),
                 (states,[0,1],{0:10,1:10,2:10},'frontier',32,None),
                 (states,[0,1],{},'frontier_editgain',32,np.ones(3)),
                 (states,[0,1],{},'frontier_editgain',32,-np.ones(6))]
        for args in invalid:
            with self.subTest(args=args[3]),self.assertRaises(ValueError):m.order_from_states(*args)

    def test_edit_potential_respects_vocab_addition_budget_and_removal_sign(self):
        m=self.module()
        x=np.array([[1,0,0,2,0],[0,0,1,0,0]],np.float32)
        g=np.array([[-3,4,5,-2,100],[2,3,-4,20,100]],np.float32)
        # Only first three columns can receive generated words; remove original
        # features over all dimensions, plus the best single absent-word addition.
        np.testing.assert_array_equal(m.edit_potential(g,x,3,1),[12,7])
        np.testing.assert_array_equal(m.edit_potential(g,x,3,2),[16,9])

    def test_surrogate_gradient_matches_finite_difference_without_mutating_model(self):
        m=self.module()
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor([[.4,-.2],[-.1,.7]],dtype=torch.float64))
            def forward(self,x,adj):return torch.log_softmax(adj@x@self.weight,dim=1)
        model=Tiny();model.train();model.weight.grad=torch.ones_like(model.weight)
        x=torch.tensor([[1.,.2],[.1,1.],[.3,.7]],dtype=torch.float64)
        adj=torch.tensor([[.5,.5,0],[.3,.4,.3],[0,.5,.5]],dtype=torch.float64)
        pseudo=torch.tensor([0,1,1]);targets=np.array([1,2])
        before=model.weight.detach().clone();old_grad=model.weight.grad.clone()
        grad,loss=m.feature_loss_gradient(model,x,adj,pseudo,targets)
        eps=1e-5;xp=x.clone();xm=x.clone();xp[0,0]+=eps;xm[0,0]-=eps
        expected=(torch.nn.functional.nll_loss(model(xp,adj)[targets],pseudo[targets])-
                  torch.nn.functional.nll_loss(model(xm,adj)[targets],pseudo[targets]))/(2*eps)
        self.assertAlmostEqual(float(grad[0,0]),float(expected.detach()),places=6)
        self.assertTrue(model.training);self.assertGreater(loss,0)
        torch.testing.assert_close(model.weight,before);torch.testing.assert_close(model.weight.grad,old_grad)
        self.assertIsNone(x.grad)

    def test_compute_states_preserves_single_graph_and_separate_source_mean(self):
        m=self.module()
        from gbhaa_experiments.post_tuning_protocol import diffusion_scores
        adj=sp.csr_matrix([[0,1,1,0,0,0],[1,0,0,1,0,0],[1,0,0,0,1,0],
                          [0,1,0,0,0,1],[0,0,1,0,0,0],[0,0,0,1,0,0]],dtype=np.float32)
        x=np.ones((6,2),np.float32);cfg=get_profile('balanced_frontier')
        states=m.compute_states(x,adj,[0,1],cfg)
        self.assertEqual(states.shape,(2,6))
        np.testing.assert_allclose(states.mean(axis=0),diffusion_scores(x,adj,[0,1],cfg),atol=1e-7)

    def test_new_schema_is_strict_and_old_scorer_cannot_silently_ignore_mechanism(self):
        self.module()
        from gbhaa_experiments.post_tuning_protocol import score_nodes
        cfg=get_profile('frontier')
        for change in ({'selector_rule':'bad'},{'selector_pool_size':3},{'selector_pool_size':257},
                       {'selector_pool_size':True}):
            with self.assertRaises(ValueError):validate_config({**cfg,**change})
        incomplete=dict(cfg);del incomplete['selector_pool_size']
        with self.assertRaises(ValueError):validate_config(incomplete)
        with self.assertRaisesRegex(ValueError,'mechanism'):
            score_nodes(np.ones((6,2)),sp.eye(6),[0,1],cfg)


if __name__=='__main__':unittest.main()
