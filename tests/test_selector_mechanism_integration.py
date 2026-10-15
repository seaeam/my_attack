"""Exercise real selector dispatch and audits, without fitting or generation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph import utils

from gbhaa_experiments.action_state_protocol import flip_action, independent_restarts
from gbhaa_experiments.entrypoints.meta_post_tuning import TuningAttack
from gbhaa_experiments.followup_protocol import graph_hash
from gbhaa_experiments.post_tuning_protocol import get_profile
from gbhaa_experiments import selector_mechanisms as mechanism


class TinySurrogate(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([[.4,-.2],[-.1,.7],[.2,.1]],dtype=torch.float32))
        self.output=self(x,torch.eye(len(x))).detach()
    def forward(self,x,adj):
        return torch.log_softmax(adj@x@self.weight,dim=1)


class MechanismIntegrationTests(unittest.TestCase):
    def setup_attack(self, profile, arm, path):
        x=torch.tensor(np.random.default_rng(8).uniform(size=(9,3)),dtype=torch.float32)
        attack=TuningAttack.__new__(TuningAttack);torch.nn.Module.__init__(attack)
        attack.config=get_profile(profile);attack.args=SimpleNamespace(seed=15)
        attack.options=SimpleNamespace(phase=arm);attack.device='cpu';attack.nnodes=len(x)
        attack.round_index=0;attack._attacked_nodes={};attack.text_attack_max_visits=10
        attack.text_attack_total_nodes=None;attack.diagnostics=[{}];attack.trace_dir=Path(path)
        attack.text_generator=SimpleNamespace(vocab=['a','b','c'])
        attack.surrogate=TinySurrogate(x)
        # Node 8 intentionally outside train/val/test: still a gradient target.
        attack.configure_evaluation_split([0,1],[2,3],[4,5,6,7])
        info=attack._prepare_selector_surrogate(x)
        return attack,x,info

    def test_real_graph_dispatch_and_tamper_rejection_all_nine_cells(self):
        from gbhaa_experiments.selector_mechanism_validation import audit_mechanism_rounds
        for profile in mechanism.RULES:
            for arm in ('endpoint_post','endpoint_pre','parallel'):
                with self.subTest(profile=profile,arm=arm), tempfile.TemporaryDirectory() as path:
                    attack,x,info=self.setup_attack(profile,arm,path)
                    pre=sp.csr_matrix(np.ones((9,9),np.float32)-np.eye(9,dtype=np.float32))
                    action=[0,1,0];post=flip_action(pre,action)
                    nodes=attack._select_round(x,pre,post,action)
                    diagnostic=attack.diagnostics[0];diagnostic.update(action=action,round=0)
                    seeds=independent_restarts(9,15,0) if arm=='parallel' else action[:2]
                    graph=post if arm=='endpoint_post' else pre
                    chosen=diagnostic['mechanism_traces']['parallel' if arm=='parallel' else ('post' if arm=='endpoint_post' else 'pre')]
                    self.assertEqual(chosen['graph_hash'],graph_hash(graph))
                    self.assertEqual(chosen['restart_nodes'],list(seeds))
                    states=mechanism.compute_states(x,graph,seeds,attack.config)
                    np.testing.assert_allclose(chosen['states'],states,atol=1e-7)
                    np.testing.assert_array_equal(attack.full_adj.to_dense().numpy(),post.toarray())
                    self.assertEqual(nodes.tolist(),chosen['selection']['selected_nodes'])
                    if profile in mechanism.GUIDED_SELECTOR_RULES:
                        np.testing.assert_array_equal(attack._selector_targets,np.arange(2,9))
                        self.assertTrue(attack.surrogate.training)
                        self.assertIsNone(attack.surrogate.weight.grad)
                        normalized=utils.normalize_adj_tensor(utils.sparse_mx_to_torch_sparse_tensor(graph),sparse=True)
                        gradient,loss=mechanism.feature_loss_gradient(attack.surrogate,x,normalized,
                            attack._selector_pseudo,attack._selector_targets)
                        expected_gain=mechanism.edit_potential(gradient,x.numpy(),3,70)
                        np.testing.assert_allclose(chosen['gains'],expected_gain)
                        self.assertAlmostEqual(chosen['gradient_loss'],loss)
                    metadata=dict(config=attack.config,variant=arm,seed=15,actions=[action],selector_surrogate=info)
                    artifacts=Path(path);(artifacts/'rounds.json').write_text(json.dumps({'rounds':[diagnostic]}))
                    features=(x.numpy(),x.numpy(),nodes[None],x.numpy()[nodes][None])
                    audit_mechanism_rounds(artifacts,metadata,pre,features,attack.splits)
                    tampered=copy.deepcopy(diagnostic)
                    key='parallel' if arm=='parallel' else 'post'
                    tampered['mechanism_traces'][key]['selection']['selected_nodes']=[0,1,2,3]
                    (artifacts/'rounds.json').write_text(json.dumps({'rounds':[tampered]}))
                    with self.assertRaises(ValueError):
                        audit_mechanism_rounds(artifacts,metadata,pre,features,attack.splits)
                    tampered=copy.deepcopy(diagnostic)
                    # A different normalized score vector must fail actual graph recomputation.
                    tampered['mechanism_traces']['pre']['states'][0]=[1/9]*9
                    (artifacts/'rounds.json').write_text(json.dumps({'rounds':[tampered]}))
                    with self.assertRaises(ValueError):
                        audit_mechanism_rounds(artifacts,metadata,pre,features,attack.splits)

    def test_almost_equal_saved_scores_cannot_change_tie_break(self):
        from gbhaa_experiments.selector_mechanism_validation import audit_mechanism_rounds
        with tempfile.TemporaryDirectory() as path:
            attack,x,info=self.setup_attack('frontier','endpoint_pre',path)
            x=torch.ones_like(x)
            pre=sp.csr_matrix(np.ones((9,9),np.float32)-np.eye(9,dtype=np.float32))
            action=[0,1,0];post=flip_action(pre,action)
            attack._select_round(x,pre,post,action)
            d=attack.diagnostics[0];d.update(action=action,round=0)
            trace=d['mechanism_traces']['pre']
            states=np.asarray(trace['states'],dtype=np.float32)
            states[0,8]=np.nextafter(states[0,8],np.float32(np.inf))
            # Float tolerance accepts this one-ULP change, but deterministic
            # ordering must be based on recomputed states, not saved evidence.
            order,_,selection=mechanism.order_from_states(states,[0,1],{},'frontier',32)
            trace['states']=states.tolist();trace['selection']=selection
            d['probe_pre_nodes']=order[:4].tolist();d['selected_nodes']=order[:4].tolist()
            post_scores=np.asarray(d['mechanism_traces']['post']['states'],dtype=np.float32).mean(axis=0)
            d['probe_score_l1']=float(np.abs(states.mean(axis=0)-post_scores).sum())
            (Path(path)/'rounds.json').write_text(json.dumps({'rounds':[d]}))
            metadata=dict(config=attack.config,variant='endpoint_pre',seed=15,actions=[action])
            nodes=order[:4];features=(x.numpy(),x.numpy(),nodes[None],x.numpy()[nodes][None])
            with self.assertRaises(ValueError):
                audit_mechanism_rounds(Path(path),metadata,pre,features,attack.splits)

    def test_real_deeprobust_surrogate_supports_sparse_graph_feature_gradients(self):
        from deeprobust.graph.defense import GCN
        with tempfile.TemporaryDirectory() as path:
            attack,x,_=self.setup_attack('frontier_editgain','endpoint_post',path)
            model=GCN(nfeat=3,nhid=16,nclass=2,dropout=.5,with_relu=False,with_bias=True,device='cpu')
            sparse=utils.sparse_mx_to_torch_sparse_tensor(sp.eye(9,format='csr'))
            normalized=utils.normalize_adj_tensor(sparse,sparse=True)
            model.eval()
            with torch.no_grad():pseudo=model(x,normalized).argmax(1)
            model.train()
            before={key:value.clone() for key,value in model.state_dict().items()}
            gradient,loss=mechanism.feature_loss_gradient(model,x,normalized,pseudo,np.arange(2,9))
            self.assertEqual(gradient.shape,x.shape);self.assertTrue(np.isfinite(gradient).all())
            self.assertGreater(loss,0);self.assertTrue(model.training)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key])

    def test_surrogate_audit_rejects_training_targets_and_changed_pseudo_labels(self):
        from gbhaa_experiments.selector_mechanism_validation import _surrogate
        with tempfile.TemporaryDirectory() as path:
            attack,x,info=self.setup_attack('frontier_editgain','endpoint_post',path)
            _surrogate(Path(path),{'selector_surrogate':info},x.numpy(),attack.splits)
            file=Path(path)/'selector_pseudolabels.npz'
            with np.load(file) as z: valid={k:z[k].copy() for k in z.files}
            for change in ({'target_nodes':np.arange(9,dtype=np.int64)},
                           {'pseudo_labels':1-valid['pseudo_labels']}):
                np.savez_compressed(file,**{**valid,**change})
                with self.assertRaises(ValueError):
                    _surrogate(Path(path),{'selector_surrogate':info},x.numpy(),attack.splits)

    def test_clean_pseudo_labels_frozen_before_graph_and_feature_changes(self):
        with tempfile.TemporaryDirectory() as path:
            attack,x,info=self.setup_attack('frontier_editgain','endpoint_post',path)
            before=attack._selector_pseudo.clone()
            attack.surrogate.output=attack.surrogate.output.flip(1)
            adj=sp.csr_matrix(np.ones((9,9),np.float32)-np.eye(9,dtype=np.float32))
            with patch.object(mechanism,'feature_loss_gradient',wraps=mechanism.feature_loss_gradient) as spy:
                attack._rank_graph(x*.9,adj,[0,1])
            np.testing.assert_array_equal(spy.call_args.args[3].numpy(),before.numpy())
            self.assertEqual(info['target_nodes_hash'],attack._selector_surrogate_info['target_nodes_hash'])


if __name__=='__main__':unittest.main()
