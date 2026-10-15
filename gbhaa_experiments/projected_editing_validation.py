"""Recompute opt-in editor artifacts without fitting or requesting generation."""
from __future__ import annotations
import hashlib
import json
import pickle
import numpy as np
import torch
from deeprobust.graph.defense import GCN
from deeprobust.graph import utils
from gbhaa_experiments.action_state_protocol import flip_action, independent_restarts
from gbhaa_experiments.action_state_validation import _same
from gbhaa_experiments.followup_protocol import array_hash
from gbhaa_experiments.projected_editing import VERSION
from gbhaa_experiments.post_tuning_protocol import score_nodes
from gbhaa_experiments.projected_editing_runtime import edit_round
from gbhaa_experiments.selector_mechanism_validation import _surrogate
from gbhaa_experiments.summarize_followup import _read_json,_require


def audit_editor(artifacts,metadata,original,features,splits):
    # Entry-point scoring uses one CPU thread. Reproduce that for exact hashes
    # even when summary is invoked from a different ambient thread setting.
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            return _audit_editor(artifacts,metadata,original,features,splits)
    finally:
        torch.set_num_threads(threads)


def _audit_editor(artifacts,metadata,original,features,splits):
    _surrogate(artifacts,metadata,features[0],splits)
    snapshot=metadata.get('editor_snapshot',{})
    _same(snapshot.get('version'),VERSION,'editor version')
    _same(snapshot.get('scoring_device'),'cpu','editor scoring device')
    model=GCN(**snapshot['architecture'],device='cpu').eval()
    with np.load(artifacts/'editor_weights.npz',allow_pickle=False) as z:weights={k:z[k] for k in z.files}
    hashes={k:array_hash(v) for k,v in weights.items()}
    _same(hashes,snapshot.get('state_hashes'),'editor saved weights')
    transfer=metadata['config'].get('editor_proxy')=='relu'
    if not transfer:
        _same(hashes,metadata['selector_surrogate']['state_hashes'],'editor vs clean surrogate weights')
        _require('transfer' not in snapshot,'Unexpected transfer snapshot')
    model.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()},strict=True)
    blob=(artifacts/'editor_vectorizer.pkl').read_bytes()
    _same(hashlib.sha256(blob).hexdigest(),snapshot.get('vectorizer_sha256'),'editor vectorizer digest')
    # These are local experiment artifacts, like the existing cached vectorizer.
    vectorizer=pickle.loads(blob)
    vocab=snapshot['vocabulary']
    _same(list(map(str,vectorizer.get_feature_names_out())),vocab,'editor vocabulary mapping')
    _same(min(len(vocab),features[0].shape[1]),metadata['selector_surrogate']['vocabulary_width'],'editor vocabulary width')
    with np.load(artifacts/'selector_pseudolabels.npz',allow_pickle=False) as z:
        pseudo=z['pseudo_labels'];targets=z['target_nodes'];clean_output=z['clean_log_probabilities']
    if transfer:
        from gbhaa_experiments.editor_transfer import audit_transfer
        audit_transfer(artifacts,metadata,model,original,features[0],splits,clean_output)
    else:
        normalized=utils.normalize_adj_tensor(utils.sparse_mx_to_torch_sparse_tensor(original),sparse=True)
        with torch.no_grad():reproduced=model(torch.tensor(features[0],dtype=torch.float32),normalized).numpy()
        _require(np.allclose(reproduced,clean_output,rtol=1e-5,atol=1e-6),'Editor checkpoint does not reproduce clean predictions')
    rounds=_read_json(artifacts/'rounds.json')['rounds']
    _same(len(rounds),len(metadata['actions']),'editor round count')
    current=features[0].copy();state=original.copy();keys=set();visits={}
    for index,(diagnostic,action) in enumerate(zip(rounds,metadata['actions'])):
        post=flip_action(state,action);graph=post if metadata['variant']=='endpoint_post' else state
        pre_order,pre_scores=score_nodes(current,state,action[:2],metadata['config'],visits)
        post_order,post_scores=score_nodes(current,post,action[:2],metadata['config'],visits)
        eligible=lambda order:[int(n) for n in order if visits.get(int(n),0)<10][:4]
        _same(eligible(pre_order),diagnostic['probe_pre_nodes'],'editor pre node selection')
        _same(eligible(post_order),diagnostic['probe_post_nodes'],'editor post node selection')
        if metadata['variant']=='parallel':
            order,_=score_nodes(current,state,independent_restarts(len(current),metadata['seed'],index),metadata['config'],visits)
        else:order=post_order if metadata['variant']=='endpoint_post' else pre_order
        _same(eligible(order),features[2][index].tolist(),'editor actual node selection')
        _same(float(np.abs(pre_scores-post_scores).sum()),diagnostic['probe_score_l1'],'editor probe score difference')
        trace=diagnostic.get('editing',{});responses=iter(trace.get('responses',[]));seen_messages=[]
        def generate(messages):
            seen_messages.append(messages)
            try:return next(responses)
            except StopIteration as exc:raise ValueError('Missing editor response') from exc
        extra={}
        if 'editor_objective' in metadata['config']:
            restarts=independent_restarts(len(current),metadata['seed'],index) if metadata['variant']=='parallel' else action[:2]
            extra['restarts']=restarts
            _same(diagnostic['restart_nodes'],list(restarts),'objective arm restarts')
        result,expected,bank=edit_round(current,graph,features[2][index],model,pseudo,targets,
            vectorizer,vocab,metadata['config'],metadata['seed'],index,generate,**extra)
        _require(next(responses,None) is None,'Extra editor responses')
        records=trace.get('requests',[]);_same(len(records),3,'editor request count')
        for slot,record in enumerate(records):
            request=dict(messages=seen_messages[slot],model=metadata['generation_model'],seed=metadata['generation_seed'],
                         temperature=metadata['generation_temperature'],max_tokens=metadata['generation_max_tokens'])
            key=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
            _same(record,dict(key=key,request=request,response=expected['responses'][slot],
                  response_hash=hashlib.sha256(expected['responses'][slot].encode()).hexdigest()),'editor request trace')
            keys.add(key)
        expected['requests']=records
        _same(trace,expected,'recomputed editor trace')
        with np.load(artifacts/'editor_candidates'/f'{index:04d}.npz',allow_pickle=False) as z:
            _require(np.array_equal(z['nodes'],expected['node_order']),'Editor candidate node order')
            _require(np.array_equal(z['candidates'],bank),'Editor actual projected candidate vectors differ')
        nodes=features[2][index]
        _require(np.array_equal(result.numpy()[nodes],features[3][index]),'Editor chosen writes differ from feature updates')
        counts=diagnostic['attribute_counts']
        _same(counts['template_attempts'],3,'editor template attempts');_same(counts['template_successes'],3,'editor successes')
        _same(counts['feature_changes'],sum(not np.allclose(current[n],result.numpy()[n]) for n in nodes),'editor actual changes')
        for n in nodes:visits[int(n)]=visits.get(int(n),0)+1
        current=result.numpy();state=post
    _same(sorted(keys),sorted(metadata['generation_request_keys']),'editor request-key coverage')
    _same(sum(metadata['request_cache'][k] for k in ('hits','misses')),3*len(rounds),'editor total logical requests')
    _require(np.array_equal(current,features[1]),'Editor final features differ')
