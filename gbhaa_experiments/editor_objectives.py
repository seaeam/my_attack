"""Label-free editor objectives and weights frozen at the start of each round."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from gbhaa_experiments.followup_protocol import array_hash, as_array
from gbhaa_experiments.post_tuning_protocol import diffusion_scores, validate_config, _restart_distribution

VERSION = 'arm-local-objective-v1'


def _target_ids(targets, n):
    ids = as_array(targets)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= n) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Objective targets must be distinct valid integer IDs')
    return ids.astype(np.int64, copy=False)


def target_weights(features, graph, restarts, targets, config):
    """All nontraining targets retain positive mass; no labels or other graph."""
    config = validate_config(config)
    ids = _target_ids(targets, len(features))
    restarts = list(restarts) if restarts is not None else []
    _restart_distribution(restarts, len(features))
    mix = config['editor_local_mix']
    uniform = np.full(len(ids), 1. / len(ids), dtype=np.float64)
    mass = None
    fallback = False
    weights = uniform
    if mix:
        local = diffusion_scores(features, graph, restarts, config)[ids].astype(np.float64)
        mass = float(local.sum())
        if not np.isfinite(local).all() or np.any(local < 0):
            raise ValueError('Invalid objective diffusion weights')
        fallback = mass <= 1e-12
        if not fallback:
            weights = (1. - mix) * uniform + mix * local / mass
    weights = weights.astype(np.float32)
    info = dict(version=VERSION, kind=config['editor_objective'], local_mix=mix,
        restart_nodes=list(map(int,restarts)), target_nodes=ids.tolist(), weights=weights.tolist(),
        weights_hash=array_hash(weights), uniform_fallback=fallback, target_diffusion_mass=mass,
        effective_target_count=float(1. / np.square(weights.astype(np.float64)).sum()),
        state='round-start; frozen for gradient and all sequential candidate scores')
    return weights, info


def objective_value(output, pseudo, targets, weights, kind):
    """Maximize weighted pseudo-NLL or margin capped at the class boundary."""
    if output.ndim != 2 or output.shape[1] < 2 or not torch.isfinite(output).all():
        raise ValueError('Objective requires finite multiclass outputs')
    ids = torch.as_tensor(_target_ids(targets, len(output)), device=output.device)
    labels = torch.as_tensor(pseudo, device=output.device)
    if (labels.dtype != torch.long or labels.shape != (len(output),)
            or torch.any(labels < 0) or torch.any(labels >= output.shape[1])):
        raise ValueError('Invalid fixed objective pseudo-labels')
    weights = torch.as_tensor(weights, dtype=output.dtype, device=output.device)
    if (weights.shape != (len(ids),) or not torch.isfinite(weights).all()
            or torch.any(weights <= 0) or not torch.isclose(weights.sum(), weights.new_tensor(1.),atol=1e-6,rtol=1e-6)):
        raise ValueError('Objective weights must be positive, finite and normalized')
    selected = output[ids]
    truth = labels[ids]
    if kind == 'nll':
        per_target = F.nll_loss(selected, truth, reduction='none')
    elif kind == 'capped_margin':
        own = selected.gather(1, truth[:,None]).squeeze(1)
        other = selected.masked_fill(F.one_hot(truth,output.shape[1]).bool(), -torch.inf).max(1).values
        per_target = -torch.relu(own - other)
    else:
        raise ValueError('Unknown editor objective')
    return (weights * per_target).sum()


def objective_gradient(model, features, normalized, pseudo, targets, weights, kind):
    """Same scalar as candidate scoring; no fitting, parameter updates or labels."""
    x = features.detach().clone().requires_grad_(True)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.enable_grad():
            value = objective_value(model(x, normalized), pseudo, targets, weights, kind)
            gradient, = torch.autograd.grad(value, x)
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite editor objective or gradient')
        return gradient.detach().cpu().numpy(), float(value.detach())
    finally:
        for module, mode in modes:
            module.training = mode
