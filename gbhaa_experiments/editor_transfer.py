"""Train-only clean ReLU proxy, independent of structural search and victim RNG."""
from __future__ import annotations

import random
import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN

from gbhaa_experiments.followup_protocol import as_array, array_hash, graph_hash, seed_stream
from gbhaa_experiments.action_state_validation import _same
from gbhaa_experiments.summarize_followup import _require

FIT = dict(train_iters=200, initialize=True, normalize=True, idx_val=None,
           optimizer='Adam', lr=.01, weight_decay=.0005, device='cpu',
           labels='training nodes only; zeros elsewhere')


def prepare_transfer(folder, source_spec, source_weights, features, graph, train, train_labels, seed):
    if features is None or graph is None or train_labels is None:
        raise ValueError('ReLU editor requires clean features/graph and training labels')
    x=np.asarray(as_array(features),dtype=np.float32)
    graph=sp.csr_matrix(graph);train=np.asarray(train,dtype=np.int64)
    y=np.asarray(train_labels)
    if (y.shape!=train.shape or y.dtype.kind not in 'iu' or len(train)==0
            or len(set(train.tolist()))!=len(train) or np.any((train<0)|(train>=len(x)))
            or np.any((y<0)|(y>=source_spec['nclass']))):
        raise ValueError('Expected valid training IDs and only their labels')
    y=y.astype(np.int64,copy=True);labels=np.zeros(len(x),dtype=np.int64);labels[train]=y
    spec={**source_spec,'with_relu':True};init_seed=seed_stream(seed,'editor-transfer')
    # GCN fitting is CPU-only. Seed only its CPU generator: torch.manual_seed
    # also changes MPS/CUDA/XPU RNGs that a CPU fork does not restore.
    py_state=random.getstate();np_state=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(init_seed);np.random.seed(init_seed)
            torch.random.default_generator.manual_seed(init_seed)
            model=GCN(**spec,lr=FIT['lr'],weight_decay=FIT['weight_decay'],device='cpu')
            model.fit(x.copy(),graph.copy(),labels,train.copy(),idx_val=None,
                      train_iters=FIT['train_iters'],initialize=True,normalize=True,verbose=False)
            model.eval()
    finally:
        random.setstate(py_state);np.random.set_state(np_state)
    clean=as_array(model.output).copy()
    np.savez_compressed(folder/'editor_source_weights.npz',**source_weights)
    np.savez_compressed(folder/'editor_transfer_inputs.npz',train_nodes=train,train_labels=y,clean_output=clean)
    receipt=dict(version='clean-relu-proxy-v1',seed=init_seed,fit=dict(FIT),source_architecture=source_spec,
        source_state_hashes={k:array_hash(v) for k,v in source_weights.items()},
        features_hash=array_hash(x),graph_hash=graph_hash(graph),train_nodes_hash=array_hash(train),
        train_labels_hash=array_hash(y),clean_output_hash=array_hash(clean))
    return model,receipt


def audit_transfer(artifacts, metadata, model, original, features, splits, source_output):
    """Check saved inputs/checkpoints/forwards. Never fit during artifact audits."""
    snapshot=metadata['editor_snapshot'];info=snapshot.get('transfer',{})
    _same(info.get('version'),'clean-relu-proxy-v1','transfer version')
    _same(info.get('fit'),FIT,'transfer fitting recipe')
    _same(info.get('seed'),seed_stream(metadata['seed'],'editor-transfer'),'transfer seed')
    _same(info.get('features_hash'),array_hash(features),'transfer clean features')
    _same(info.get('graph_hash'),graph_hash(original),'transfer clean graph')
    _same(snapshot['architecture'],{**info['source_architecture'],'with_relu':True},'transfer architecture')
    _same(info['source_state_hashes'],metadata['selector_surrogate']['state_hashes'],'transfer source identity')
    with np.load(artifacts/'editor_source_weights.npz',allow_pickle=False) as z:
        weights={k:z[k] for k in z.files}
    _same({k:array_hash(v) for k,v in weights.items()},info['source_state_hashes'],'transfer source weights')
    source=GCN(**info['source_architecture'],device='cpu').eval()
    source.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()},strict=True)
    with np.load(artifacts/'editor_transfer_inputs.npz',allow_pickle=False) as z:
        train=z['train_nodes'];labels=z['train_labels'];clean=z['clean_output']
    _require(np.array_equal(train,splits['train']),'Transfer training split differs')
    _require(labels.dtype.kind in 'iu' and labels.shape==train.shape
        and np.all((labels>=0)&(labels<snapshot['architecture']['nclass'])),'Invalid transfer training labels')
    for values,key in ((train,'train_nodes_hash'),(labels,'train_labels_hash'),(clean,'clean_output_hash')):
        _same(array_hash(values),info[key],f'transfer {key}')
    norm=utils.normalize_adj_tensor(utils.sparse_mx_to_torch_sparse_tensor(original),sparse=True)
    x=torch.tensor(features,dtype=torch.float32)
    with torch.no_grad():old=source(x,norm).numpy();new=model(x,norm).numpy()
    _require(np.allclose(old,source_output,rtol=1e-5,atol=1e-6),'Source checkpoint does not reproduce pseudo-label origin')
    _require(np.allclose(new,clean,rtol=1e-5,atol=1e-6),'Transfer checkpoint does not reproduce clean output')
