"""Real GCN forward/backward and saved audits; no fitting or network generation."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from sklearn.feature_extraction.text import CountVectorizer
from gbhaa_experiments.entrypoints.meta_post_tuning import TuningAttack
from gbhaa_experiments.post_tuning_protocol import get_profile,EDIT_PROFILES
from gbhaa_experiments.projected_editing_runtime import prepare_editor,edit_round
from gbhaa_experiments.projected_editing_validation import audit_editor
from gbhaa_experiments.action_state_protocol import flip_action
from gbhaa_experiments.action_state_cache import RequestCache
from gbhaa_experiments.followup_protocol import graph_hash,seed_stream,array_hash

class EditingIntegrationTests(unittest.TestCase):
    def setup_attack(self,path,profile,arm):
        torch.manual_seed(12);torch.set_num_threads(1)
        x=torch.tensor(np.random.default_rng(3).integers(0,2,(9,8)),dtype=torch.float32)
        pre=sp.csr_matrix(np.ones((9,9),np.float32)-np.eye(9,dtype=np.float32))
        attack=TuningAttack.__new__(TuningAttack);torch.nn.Module.__init__(attack)
        attack.config=get_profile(profile);attack.args=SimpleNamespace(seed=15)
        attack.options=SimpleNamespace(phase=arm);attack.device='cpu';attack.nnodes=9
        attack.round_index=0;attack._attacked_nodes={};attack.text_attack_max_visits=10
        attack.text_attack_total_nodes=None;attack.diagnostics=[{}];attack.trace_dir=Path(path);attack.attribute_counts={}
        vocab=['apple','berry','cherry','dates','elder','figs','grape','honey']
        calls=[]
        def generate(messages):calls.append(messages);return ['apple cherry','berry dates','grape figs'][len(calls)%3]
        attack.text_generator=SimpleNamespace(vocab=vocab,vectorizer=CountVectorizer(vocabulary=vocab,binary=True),
            generate_text=generate,model_name='fixture',generation_seed=seed_stream(15,'generation'),generation_temperature=0.,max_tokens=50)
        attack.request_cache=RequestCache(Path(path)/'cache');attack.request_cache.attach(attack.text_generator)
        attack.surrogate=GCN(nfeat=8,nhid=5,nclass=2,with_relu=False,device='cpu').eval()
        norm=utils.normalize_adj_tensor(utils.sparse_mx_to_torch_sparse_tensor(pre),sparse=True)
        with torch.no_grad():attack.surrogate.output=attack.surrogate(x,norm)
        attack.configure_evaluation_split([0,1],[2,3],[4,5,6,7])
        info=attack._prepare_selector_surrogate(x);snapshot=prepare_editor(attack)
        return attack,x,pre,info,snapshot,calls
    def fixture(self,path,profile,arm,round_count=1):
        a,x,pre,info,snapshot,calls=self.setup_attack(path,profile,arm)
        orig=x.numpy().copy();state=pre;updates=[];ids=[];actions=[]
        for i in range(round_count):
            if i:a.diagnostics.append({})
            a.round_index=i;action=[0,i+1,0];post=flip_action(state,action)
            nodes=a._select_round(x,state,post,action)
            self.assertEqual(graph_hash(a._editor_graph),graph_hash(post if arm=='endpoint_post' else state))
            a.diagnostics[-1].update(action=action,round=i)
            x=a.attack_features_with_text(target_nodes=nodes,full_features=x,budget_per_node=80,text_clusters={0:nodes})
            for n in nodes:a._attacked_nodes[int(n)]=a._attacked_nodes.get(int(n),0)+1
            updates.append(x.numpy()[nodes].copy());ids.append(nodes);actions.append(action);state=post
        metadata=dict(config=a.config,variant=arm,seed=15,actions=actions,selector_surrogate=info,editor_snapshot=snapshot,
            generation_request_keys=sorted(a.request_cache.used_keys),request_cache=a.request_cache.stats,
            generation_model='fixture',generation_seed=seed_stream(15,'generation'),generation_temperature=0.,generation_max_tokens=50)
        features=(orig,x.numpy(),np.stack(ids),np.stack(updates))
        (Path(path)/'rounds.json').write_text(json.dumps({'rounds':a.diagnostics}))
        return a,metadata,pre,features,calls
    def test_all_twelve_dispatches_and_audits(self):
        for profile in EDIT_PROFILES:
            for arm in ('endpoint_post','endpoint_pre','parallel'):
                with self.subTest(profile=profile,arm=arm),tempfile.TemporaryDirectory() as path:
                    a,m,g,f,calls=self.fixture(path,profile,arm)
                    self.assertEqual(len(calls),3);self.assertEqual(a.attribute_counts['feature_writes'],4)
                    self.assertTrue(np.any(f[0]!=f[1]));audit_editor(Path(path),m,g,f,a.splits)
                    t=a.diagnostics[0]['editing']
                    self.assertEqual(t['features_hash'],array_hash(f[0]))
                    for before,after in zip(t['nodes'],t['nodes'][1:]):
                        self.assertEqual(after['base_loss'],before['candidate_losses'][before['chosen']])
                    self.assertAlmostEqual(t['actual_gain'],sum(n['actual_gain'] for n in t['nodes']))
                    self.assertTrue(any(n['base_loss']!=t['initial_loss'] for n in t['nodes'][1:]))
                    if a.config['editor_gain']:
                        for n in t['nodes']:self.assertEqual(n['chosen'],int(np.argmax(n['candidate_losses'])))
    def test_tamper_graph_loss_word_plan_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as path:
            a,m,g,f,_=self.fixture(path,'edit_both','endpoint_pre',2)
            audit_editor(Path(path),m,g,f,a.splits)
            for mutation in ('graph','loss','plan','response','choice'):
                d=copy.deepcopy(a.diagnostics);t=d[0]['editing']
                if mutation=='graph':t['graph_hash']='0'*64
                if mutation=='loss':t['nodes'][1]['candidate_losses'][0]+=.1
                if mutation=='plan':t['nodes'][0]['plans'][0]['drop']=[0,1]
                if mutation=='response':t['responses'][0]+=' honey'
                if mutation=='choice':t['nodes'][0]['chosen']=(t['nodes'][0]['chosen']+1)%3
                (Path(path)/'rounds.json').write_text(json.dumps({'rounds':d}))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit_editor(Path(path),m,g,f,a.splits)
            (Path(path)/'rounds.json').write_text(json.dumps({'rounds':a.diagnostics}))
            file=Path(path)/'editor_weights.npz'
            with np.load(file) as z:weights={k:z[k] for k in z.files}
            weights['gc1.weight'][0,0]+=.1;np.savez_compressed(file,**weights)
            with self.assertRaises(ValueError):audit_editor(Path(path),m,g,f,a.splits)
    def test_choice_factor_does_not_change_generation_at_same_start(self):
        for pair in (('edit_control','edit_gain'),('edit_words','edit_both')):
            with tempfile.TemporaryDirectory() as p,tempfile.TemporaryDirectory() as q:
                a,*_=self.fixture(p,pair[0],'endpoint_post');b,*_=self.fixture(q,pair[1],'endpoint_post')
                self.assertEqual(a.diagnostics[0]['editing']['messages'],b.diagnostics[0]['editing']['messages'])
                for x,y in zip(a.diagnostics[0]['editing']['nodes'],b.diagnostics[0]['editing']['nodes']):
                    self.assertEqual(x['candidate_hashes'],y['candidate_hashes'])

    def test_snapshot_handles_nonleaf_clean_outputs_without_copying_autograd(self):
        with tempfile.TemporaryDirectory() as path:
            a,x,g,*_=self.setup_attack(path,'edit_control','endpoint_post')
            norm=utils.normalize_adj_tensor(utils.sparse_mx_to_torch_sparse_tensor(g),sparse=True)
            a.surrogate.output=a.surrogate(x,norm)
            self.assertIsNotNone(a.surrogate.output.grad_fn)
            state=torch.random.get_rng_state().clone()
            prepare_editor(a)
            self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_identical_real_responses_are_not_resampled(self):
        with tempfile.TemporaryDirectory() as path:
            a,x,g,*_=self.setup_attack(path,'edit_control','endpoint_post')
            calls=[]
            def generate(messages):calls.append(messages);return 'apple berry'
            _,trace,bank=edit_round(x,g,[0,1,2,3],a._editor_model,a._selector_pseudo,a._selector_targets,
                a.text_generator.vectorizer,a.text_generator.vocab,a.config,15,0,generate)
            self.assertEqual(len(calls),3);self.assertEqual(trace['distinct_responses'],1)
            self.assertEqual(bank.shape,(4,3,8))
            for n in trace['nodes']:self.assertEqual(len(n['candidate_losses']),3)

    def test_all_negative_candidate_gains_still_write_best_real_candidate(self):
        with tempfile.TemporaryDirectory() as path:
            a,x,g,*_=self.setup_attack(path,'edit_gain','endpoint_post')
            x.zero_();x[:,0]=1
            with torch.no_grad():
                a._editor_model.gc1.weight.fill_(.1);a._editor_model.gc1.bias.zero_()
                a._editor_model.gc2.weight[:,0]=.2;a._editor_model.gc2.weight[:,1]=-.2
                a._editor_model.gc2.bias.zero_()
            result,trace,_=edit_round(x,g,[0,1,2,3],a._editor_model,torch.zeros(9,dtype=torch.long),np.arange(2,9),
                a.text_generator.vectorizer,a.text_generator.vocab,a.config,15,0,lambda _: 'apple berry')
            self.assertTrue(all(max(n['candidate_losses'])<n['base_loss'] for n in trace['nodes']))
            self.assertTrue(all(n['chosen']==0 for n in trace['nodes']))
            self.assertGreater(torch.count_nonzero(result),torch.count_nonzero(x))
