"""Exercise the public loader, cache audit and report writer with offline artifacts."""
import json,unittest
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from sklearn.feature_extraction.text import CountVectorizer
import test_post_tuning_summary as legacy
from gbhaa_experiments.entrypoints.meta_post_tuning import TuningAttack
from gbhaa_experiments.post_tuning_protocol import EDIT_PROFILES
from gbhaa_experiments.projected_editing_runtime import prepare_editor
from gbhaa_experiments.action_state_cache import RequestCache
from gbhaa_experiments.action_state_protocol import flip_action,feature_measures
from gbhaa_experiments.followup_protocol import array_hash,seed_stream
from gbhaa_experiments.run_followup import fingerprint

class EditingEndToEndTests(unittest.TestCase):
    def fixture(self, profiles=EDIT_PROFILES):
        helper=legacy.PostTuningSummaryTests();helper.setUp();self.addCleanup(helper.doCleanups)
        rows=helper.fixture(seeds=(15,16),profiles=profiles,matched=True)
        for row in rows:
            artifacts=helper.directory(row)/'artifacts';m=row['attack']
            with np.load(artifacts/'features.npz') as z:original=z['original'].copy()
            graph=sp.load_npz(artifacts/'original_adj.npz');n,d=original.shape
            torch.manual_seed(row['seed']);torch.set_num_threads(1)
            a=TuningAttack.__new__(TuningAttack);torch.nn.Module.__init__(a)
            a.config=row['config'];a.args=SimpleNamespace(seed=row['seed'],hidden=5,dropout=.5);a.options=SimpleNamespace(phase=row['arm'])
            a.nnodes=n;a.device='cpu';a.trace_dir=artifacts;a._attacked_nodes={};a.text_attack_max_visits=10
            a.text_attack_total_nodes=None;a.diagnostics=[];a.attribute_counts={}
            vocab=['word'+chr(97+i) for i in range(d)]
            a.text_generator=SimpleNamespace(vocab=vocab,vectorizer=CountVectorizer(vocabulary=vocab,binary=True),
                generate_text=lambda messages:'worda wordb wordc',model_name='fixture',max_tokens=50,
                generation_seed=seed_stream(row['seed'],'generation'),generation_temperature=0.)
            a.request_cache=RequestCache(helper.root/'_generation_cache'/fingerprint(row['provenance']))
            a.request_cache.attach(a.text_generator)
            a.surrogate=GCN(nfeat=d,nhid=5,nclass=2,with_relu=False,device='cpu').eval()
            x=torch.from_numpy(original.copy());norm=utils.normalize_adj_tensor(utils.sparse_mx_to_torch_sparse_tensor(graph),sparse=True)
            a.surrogate.output=a.surrogate(x,norm)  # Deliberately nonleaf, as after the real fit.
            a.configure_evaluation_split(np.arange(6,12),np.arange(3),np.arange(3,6))
            m['selector_surrogate']=a._prepare_selector_surrogate(x)
            m['editor_snapshot']=prepare_editor(a,features=x,graph=graph,train_labels=np.arange(6)%2)
            state=graph;ids=[];updates=[]
            for i,action in enumerate(m['actions']):
                a.round_index=i;a.diagnostics.append(dict(round=i,action=action));post=flip_action(state,action)
                nodes=a._select_round(x,state,post,action)
                x=a.attack_features_with_text(target_nodes=nodes,full_features=x,budget_per_node=80,text_clusters={0:nodes})
                for node in nodes:a._attacked_nodes[int(node)]=a._attacked_nodes.get(int(node),0)+1
                ids.append(nodes);updates.append(x.numpy()[nodes].copy());state=post
            np.savez_compressed(artifacts/'features.npz',original=original,modified=x.numpy())
            np.savez_compressed(artifacts/'feature_updates.npz',nodes=np.stack(ids),values=np.stack(updates))
            (artifacts/'rounds.json').write_text(json.dumps({'rounds':a.diagnostics}))
            m.update(modified_features_hash=array_hash(x),**feature_measures(original,x.numpy()),
                attribute_node_visits=sum(a._attacked_nodes.values()),visited_attribute_nodes=len(a._attacked_nodes),
                attribute_counts=a.attribute_counts,generation_request_keys=sorted(a.request_cache.used_keys),request_cache=a.request_cache.stats)
            helper.predictions(row,correct={'edit_control':2,'edit_gain':1,'edit_words':1,'edit_both':0,
                'transfer_relu':1,'transfer_residual':1,'transfer_both':0,
                'objective_margin':1,'objective_local':1,'objective_both':0}[row['profile_id']])
            helper.save(row)
        return helper,rows
    def test_full_matrix_loads_and_summary_rejects_missing_or_altered_cache(self):
        helper,rows=self.fixture()
        for row in rows:helper.validate(row)
        code,report,recommendation=helper.summarize()
        self.assertEqual(code,0);self.assertEqual(report['status'],'complete')
        self.assertEqual(len(report['factorial_effects']),9);self.assertEqual(len(report['ranking']),4)
        self.assertEqual(recommendation['recommendations'][0]['profile_id'],'edit_both')
        self.assertTrue((helper.output/'factorial_effects.csv').exists())
        row=rows[0];key=row['attack']['generation_request_keys'][0]
        cache=helper.root/'_generation_cache'/fingerprint(row['provenance'])/(key+'.json')
        original=cache.read_text();record=json.loads(original);record['response']='altered response'
        import hashlib
        record['response_hash']=hashlib.sha256(record['response'].encode()).hexdigest();cache.write_text(json.dumps(record))
        with self.assertRaises(ValueError):helper.validate(row)
        cache.write_text(original)
        (helper.directory(rows[-1])/'result.json').unlink()
        code,report,recommendation=helper.summarize()
        self.assertEqual(code,2);self.assertFalse(recommendation['recommendations'])
        self.assertTrue(all(v['mean_pp'] is None for v in report['factorial_effects']))
