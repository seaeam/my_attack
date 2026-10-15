"""Three-request, four-write editor and self-contained fixed-model snapshots."""
from __future__ import annotations
import hashlib
import json
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from gbhaa_experiments.followup_protocol import as_array,array_hash,graph_hash
from gbhaa_experiments.projected_editing import (
    VERSION,word_plan,adapt_candidate,choose_candidate,common_order,messages_for)
from gbhaa_experiments.selector_mechanisms import feature_loss_gradient


def prepare_editor(attack, features=None, graph=None, train_labels=None):
    model=attack.surrogate
    if not isinstance(model,GCN):raise ValueError('Projected editor requires the existing fixed GCN surrogate')
    # CPU scoring makes saved audits independent of the training device.
    spec=dict(nfeat=model.nfeat,nhid=model.hidden_sizes[0],nclass=model.nclass,
              dropout=model.dropout,with_relu=model.with_relu,with_bias=model.with_bias)
    weights={k:as_array(v).copy() for k,v in model.state_dict().items()}
    transfer=None
    if attack.config.get('editor_proxy')=='relu':
        from gbhaa_experiments.editor_transfer import prepare_transfer
        if features is None or graph is None or train_labels is None:
            raise ValueError('ReLU editor requires clean features/graph and training labels')
        # The source bootstrap hardcodes hidden/dropout; the victim uses args.
        # This fixed pilot must fail rather than silently compare unlike models.
        victim=dict(nfeat=as_array(features).shape[1],nhid=getattr(attack.args,'hidden',None),
            nclass=getattr(attack,'nclass',int(np.max(train_labels))+1),
            dropout=getattr(attack.args,'dropout',None),with_relu=True,with_bias=True)
        mismatches=[k for k in victim if {**spec,'with_relu':True}[k]!=victim[k]]
        if mismatches:
            raise ValueError('Transfer pilot source differs from victim architecture: '+', '.join(mismatches))
        model,transfer=prepare_transfer(attack.trace_dir,spec,weights,features,graph,
            attack.splits['train'],train_labels,attack.args.seed)
        spec={**spec,'with_relu':True}
        weights={k:as_array(v).copy() for k,v in model.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        attack._editor_model=GCN(**spec,device='cpu').eval()
    attack._editor_model.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()})
    np.savez_compressed(attack.trace_dir/'editor_weights.npz',**weights)
    vectorizer=pickle.dumps(attack.text_generator.vectorizer,protocol=4)
    (attack.trace_dir/'editor_vectorizer.pkl').write_bytes(vectorizer)
    snapshot=dict(version=VERSION,architecture=spec,scoring_device='cpu',
                state_hashes={k:array_hash(v) for k,v in weights.items()},
                vectorizer_sha256=hashlib.sha256(vectorizer).hexdigest(),
                vocabulary=list(map(str,attack.text_generator.vocab)))
    if transfer is not None:snapshot['transfer']=transfer
    return snapshot


def request_record(generator,messages,response):
    request=dict(messages=messages,model=generator.model_name,seed=generator.generation_seed,
                 temperature=generator.generation_temperature,max_tokens=generator.max_tokens)
    key=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    return dict(key=key,request=request,response=response,response_hash=hashlib.sha256(response.encode()).hexdigest())


def edit_round(features,graph,nodes,model,pseudo,targets,vectorizer,vocab,config,seed,round_index,generate,*,restarts=None):
    """No fitting or true labels. generate(messages) returns one real response."""
    x=torch.tensor(as_array(features),dtype=torch.float32,device='cpu')
    normalized=utils.normalize_adj_tensor(utils.sparse_mx_to_torch_sparse_tensor(graph),sparse=True)
    pseudo=torch.as_tensor(as_array(pseudo),dtype=torch.long);targets=torch.as_tensor(targets,dtype=torch.long)
    objective=None
    if 'editor_objective' in config:
        from gbhaa_experiments.editor_objectives import target_weights, objective_gradient, objective_value
        weights,objective=target_weights(x,graph,restarts,targets,config)
        gradient,initial=objective_gradient(model,x,normalized,pseudo,targets,weights,config['editor_objective'])
    else:
        gradient,initial=feature_loss_gradient(model,x,normalized,pseudo,targets)
    source=x.numpy().copy();nodes=sorted(map(int,nodes));width=min(len(vocab),x.shape[1])
    if len(nodes)!=4 or len(set(nodes))!=4:raise ValueError('Editor requires four distinct nodes')
    k=config['editor_candidates'];common=common_order(source,nodes,width)
    caps=[math.ceil(config['text_cdl_topk']*(slot+1)/k) for slot in range(k)]
    plans=[{n:word_plan(source[n],gradient[n],common,cap,config['editor_gradient'],
             config['text_budget_per_node'],width) for n in nodes} for cap in caps]
    responses=[];messages=[]
    for slot in range(k):
        msg=messages_for(plans[slot],vocab,slot);reply=generate(msg)
        if not isinstance(reply,str) or not reply.strip():raise ValueError('Empty editor candidate response')
        # We ask for one candidate. Unexpected delimiters do not create bonus candidates.
        parts=[p.strip() for p in reply.split('|||') if p.strip()]
        if len(parts)!=1:raise ValueError('Editor requires exactly one nonempty template per request')
        responses.append(reply);messages.append(msg)
    modes=[(module,module.training) for module in model.modules()]
    def loss(value):
        with torch.no_grad():
            output=model(value,normalized)
            answer=float(objective_value(output,pseudo,targets,weights,config['editor_objective'])) if objective is not None else float(F.nll_loss(output[targets],pseudo[targets]))
        if not math.isfinite(answer):raise ValueError('Nonfinite projected candidate loss')
        return answer
    node_traces=[];candidates=[]
    try:
        model.eval();initial=loss(x)
        for node in nodes:
            before=loss(x);bank=[];details=[];losses=[]
            for slot,text in enumerate(responses):
                value,info=adapt_candidate(text,plans[slot][node],source[node],vectorizer,vocab,
                                           config['text_max_added_words'],config['text_similarity_min'],
                                           residual=config.get('editor_residual',False))
                bank.append(value);details.append(info)
                trial=x.clone();trial[node]=torch.from_numpy(value);losses.append(loss(trial))
            chosen=choose_candidate(losses,config['editor_gain'],seed,round_index,node)
            x[node]=torch.from_numpy(bank[chosen]);candidates.append(np.stack(bank))
            node_traces.append(dict(node=node,plans=[p[node] for p in plans],projections=details,
                candidate_hashes=[array_hash(v) for v in bank],distinct_candidates=len({array_hash(v) for v in bank}),
                base_loss=before,candidate_losses=losses,chosen=chosen,actual_gain=losses[chosen]-before))
        final=loss(x)
    finally:
        for module,mode in modes:module.training=mode
    trace=dict(version=VERSION,graph_hash=graph_hash(graph),features_hash=array_hash(source),
        node_order=nodes,word_caps=caps,gradient_hash=array_hash(gradient),initial_loss=initial,final_loss=final,
        actual_gain=final-initial,messages=messages,responses=responses,
        distinct_responses=len(set(responses)),nodes=node_traces)
    if 'editor_proxy' in config:
        trace.update(policy_version='transfer-editor-v1',proxy=config['editor_proxy'],
            edit_policy='planned_residual' if config['editor_residual'] else 'full_replacement')
    if objective is not None:
        trace['objective']=objective
    return x,trace,np.stack(candidates)


def attack_features(attack,target_nodes,full_features,labels_st=None,budget_per_node=80,
                    target_embeddings=None,text_clusters=None):
    if budget_per_node!=attack.config['text_budget_per_node']:raise ValueError('Editor word budget mismatch')
    if text_clusters is not None:
        clusters=[sorted(map(int,values)) for values in text_clusters.values() if len(values)]
        if clusters!=[sorted(map(int,target_nodes))]:raise ValueError('Editor requires one shared four-node text group')
    records=[]
    def generate(messages):
        response=attack.text_generator.generate_text(messages)
        records.append(request_record(attack.text_generator,messages,response));return response
    extra=dict(restarts=attack.diagnostics[-1]['restart_nodes']) if 'editor_objective' in attack.config else {}
    result,trace,bank=edit_round(full_features,attack._editor_graph,target_nodes,attack._editor_model,
        attack._selector_pseudo,attack._selector_targets,attack.text_generator.vectorizer,
        attack.text_generator.vocab,attack.config,attack.args.seed,attack.round_index,generate,**extra)
    trace['requests']=records
    attack.diagnostics[-1]['editing']=trace
    folder=attack.trace_dir/'editor_candidates';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{attack.round_index:04d}.npz',nodes=np.array(trace['node_order']),candidates=bank)
    changes=sum(not np.allclose(as_array(full_features)[n],as_array(result)[n]) for n in target_nodes)
    attack.finish_text_attack_round(dict(targets=4,feature_writes=4,feature_changes=changes,
        template_attempts=3,template_successes=3,template_failures=0,node_failures=0,
        cluster_failures=0,cache_hits=0,completed=int(changes>0)))
    return result.to(full_features.device)
