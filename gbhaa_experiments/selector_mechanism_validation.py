"""Audit new selector variants without fitting models or reading held-out labels."""
from __future__ import annotations

import numpy as np

from gbhaa_experiments.action_state_protocol import flip_action, independent_restarts
from gbhaa_experiments.followup_protocol import array_hash, graph_hash
from gbhaa_experiments.selector_mechanisms import compute_states, order_from_states, selector_version, GUIDED_SELECTOR_RULES
from gbhaa_experiments.summarize_followup import _hash, _object, _read_json, _require
from gbhaa_experiments.action_state_validation import _same


def _surrogate(artifacts, metadata, features, splits):
    info = _object(metadata.get('selector_surrogate'), 'selector_surrogate')
    expected = ('model_class', 'state_hashes', 'clean_output_hash', 'pseudo_labels_hash',
                'target_nodes_hash', 'nclass', 'vocabulary_width', 'loss', 'edit_proxy')
    _same(sorted(info), sorted(expected), 'selector_surrogate fields')
    _require(isinstance(info['model_class'], str) and bool(info['model_class']), 'Missing surrogate class')
    hashes = _object(info['state_hashes'], 'surrogate state hashes')
    _require(bool(hashes), 'Missing surrogate weights identity')
    for name, value in hashes.items():
        _require(isinstance(name, str) and bool(name), 'Invalid surrogate weight name')
        _hash(value, 'surrogate weight digest')
    n, d = features.shape
    _require(type(info['nclass']) is int and info['nclass'] >= 2, 'Invalid selector class count')
    _require(type(info['vocabulary_width']) is int and 0 < info['vocabulary_width'] <= d, 'Invalid selector vocabulary width')
    _same(info['loss'], 'mean NLL of fixed clean argmax pseudo-labels over all non-training nodes', 'selector loss')
    _same(info['edit_proxy'], 'beneficial removals plus bounded unit additions in absent vocabulary columns', 'selector edit proxy')
    with np.load(artifacts/'selector_pseudolabels.npz', allow_pickle=False) as data:
        pseudo, targets, output = data['pseudo_labels'], data['target_nodes'], data['clean_log_probabilities']
    _require(pseudo.shape == (n,) and pseudo.dtype.kind in 'iu', 'Invalid selector pseudo-labels')
    _require(targets.dtype.kind in 'iu' and np.array_equal(targets, np.setdiff1d(np.arange(n), splits['train'])),
             'Selector targets must be exactly all non-training nodes, including unassigned nodes')
    _require(output.shape == (n, info['nclass']) and output.dtype.kind == 'f' and np.isfinite(output).all()
             and np.all(output <= 1e-6) and np.allclose(np.exp(output).sum(axis=1), 1, atol=1e-5),
             'Invalid selector clean log probabilities')
    _require(np.array_equal(pseudo, output.argmax(axis=1)), 'Selector pseudo-labels differ from fixed clean predictions')
    for array, key in ((pseudo, 'pseudo_labels_hash'), (targets, 'target_nodes_hash'), (output, 'clean_output_hash')):
        _same(array_hash(array), info[key], key)


def audit_mechanism_rounds(artifacts, metadata, original, features, splits):
    """Recompute diffusion and ranking; gradient values are recorded evidence.

    Audits do not refit/reconstruct the surrogate. Its fixed identity, pseudo
    labels and target set are checked here and matched across arms in summaries.
    """
    config = metadata['config']
    if 'selector_rule' not in config:
        return
    guided = config['selector_rule'] in GUIDED_SELECTOR_RULES
    if guided:
        _surrogate(artifacts, metadata, features[0], splits)
    else:
        _require(metadata.get('selector_surrogate') is None, 'Unguided selector has surrogate metadata')
    rounds = _read_json(artifacts/'rounds.json')['rounds']
    _require(len(rounds) == len(metadata['actions']), 'Mechanism round count differs from actions')
    current = features[0].copy(); state = original.copy()
    visits = np.zeros(len(current), dtype=np.int64)
    parallel = metadata['variant'] == 'parallel'
    for index, (diagnostic, action) in enumerate(zip(rounds, metadata['actions'])):
        post = flip_action(state, action)
        traces = _object(diagnostic.get('mechanism_traces'), 'mechanism_traces')
        _same(sorted(traces), sorted(('pre', 'post', 'parallel') if parallel else ('pre', 'post')), 'mechanism trace keys')
        score_vectors = {}
        for name, trace in traces.items():
            trace = _object(trace, f'mechanism {name}')
            _same(sorted(trace), sorted(('version', 'rule', 'graph_hash', 'restart_nodes', 'states',
                                        'gains', 'gradient_loss', 'selection')), 'mechanism trace fields')
            graph = post if name == 'post' else state
            seeds = independent_restarts(len(current), metadata['seed'], index) if name == 'parallel' else action[:2]
            _same(trace['version'], selector_version(config['selector_rule']), 'mechanism version')
            _same(trace['rule'], config['selector_rule'], 'mechanism rule')
            _same(trace['graph_hash'], graph_hash(graph), 'mechanism graph')
            _same(trace['restart_nodes'], list(seeds), 'mechanism restarts')
            states = np.asarray(trace['states'], dtype=np.float32)
            expected_states = compute_states(current, graph, seeds, config)
            _require(states.shape == expected_states.shape and np.isfinite(states).all()
                     and np.allclose(states, expected_states, rtol=1e-6, atol=1e-9),
                     'Mechanism diffusion states differ from current features and arm graph')
            if guided:
                loss = trace['gradient_loss']
                _require(type(loss) in (int, float) and np.isfinite(loss) and loss >= 0, 'Invalid gradient loss')
            else:
                _require(trace['gains'] is None and trace['gradient_loss'] is None, 'Unguided selector contains gains')
            # Approximate numeric agreement does not preserve discrete tie-breaks.
            # Derive the authoritative candidate pool/order from recomputation.
            order, scores, info = order_from_states(expected_states, seeds, visits, config['selector_rule'],
                                                    config['selector_pool_size'], trace['gains'])
            _same(trace['selection'], info, 'mechanism selection audit')
            field = {'pre': 'probe_pre_nodes', 'post': 'probe_post_nodes', 'parallel': 'selected_nodes'}[name]
            _same(order[:4].tolist(), diagnostic[field], f'mechanism {field}')
            score_vectors[name] = scores
        actual = 'parallel' if parallel else ('post' if metadata['variant'] == 'endpoint_post' else 'pre')
        selected = traces[actual]['selection']['selected_nodes']
        _same(selected, diagnostic['selected_nodes'], 'mechanism actual selected nodes')
        _same(selected, features[2][index].tolist(), 'mechanism saved feature writes')
        _same(float(np.abs(score_vectors['pre']-score_vectors['post']).sum()), diagnostic['probe_score_l1'], 'mechanism probe score difference')
        current[selected] = features[3][index]; visits[selected] += 1; state = post
