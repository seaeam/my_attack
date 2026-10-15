"""Small, testable primitives for the opt-in AE-PR follow-up protocol."""
from __future__ import annotations

import hashlib
import json
import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from gbhaa_experiments.followup_config import PROTOCOL, SELECTOR_PROTOCOL, PR_TOL, PR_MAX_ITER


def seed_stream(seed, stream, step=0):
    raw = f"{seed}:{stream}:{step}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little") % (2**31 - 1)


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_generation(generator, seed):
    """Keep retries in our counted loop, including for an OpenAI-compatible API."""
    generator.generation_seed = seed
    generator.generation_temperature = 0.0
    generator.client = generator.client.with_options(max_retries=0)


def as_array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.is_sparse:
            value = value.to_dense()
        return value.numpy()
    return np.asarray(value)


def array_hash(value):
    arr = np.ascontiguousarray(as_array(value))
    digest = hashlib.sha256()
    digest.update(str((arr.shape, str(arr.dtype))).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def graph_hash(adj):
    csr = sp.csr_matrix(adj, dtype=np.float64, copy=True)
    csr.eliminate_zeros(); csr.sort_indices()
    payload = [csr.shape, array_hash(csr.indptr.astype(np.int64)),
               array_hash(csr.indices.astype(np.int64)), array_hash(csr.data)]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def stable_order(scores):
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite selector scores")
    return np.lexsort((np.arange(len(scores)), -scores))


def selector_order(attack, features, adj, endpoints, strategy, alpha, iterations, seed):
    """Rank the SAME full node universe; feasibility filtering follows ranking."""
    n = adj.shape[0]
    if strategy in ("ae_pr", "ppr"):
        _, scores = attack.ppr_topk_from_seeds(
            fea=features, adj=adj, seed_nodes=endpoints, topk=n,
            alpha=alpha, N=iterations, use_cos_sim=strategy == "ae_pr")
    elif strategy == "pr":
        # alpha is the restart probability here; NetworkX alpha is damping.
        # Retain all nodes, including isolates; dangling mass restarts uniformly.
        topology = sp.csr_matrix(adj, dtype=np.float64, copy=True)
        topology.setdiag(0)
        topology.eliminate_zeros()
        graph = nx.from_scipy_sparse_array(topology, create_using=nx.DiGraph)
        ranks = nx.pagerank(graph, alpha=1.0 - alpha, personalization=None,
                            dangling=None, weight=None, tol=PR_TOL, max_iter=PR_MAX_ITER)
        scores = np.array([ranks[node] for node in range(n)], dtype=np.float64)
    elif strategy == "degree":
        scores = np.asarray(adj.sum(axis=1)).ravel()
    elif strategy == "random":
        scores = np.random.default_rng(seed).random(n)
    else:
        raise ValueError(f"Unknown matched selector: {strategy}")
    return stable_order(scores), scores


def prediction_metrics(clean, attacked, labels):
    clean, attacked, labels = map(np.asarray, (clean, attacked, labels))
    if clean.shape != attacked.shape or labels.shape != clean.shape or not clean.size:
        raise ValueError("Predictions and labels must share a nonempty target set")
    correct = clean == labels
    count = int(correct.sum())
    flips = int(np.count_nonzero(correct & (attacked != labels)))
    ca, aa = float(correct.mean()), float((attacked == labels).mean())
    return {"clean_accuracy": ca, "attacked_accuracy": aa,
            "accuracy_drop_pp": 100 * (ca - aa), "misclassification": 1 - aa,
            "clean_correct_count": count, "attack_flips": flips,
            "conditional_asr": flips / count if count else None,
            "target_count": int(labels.size)}
