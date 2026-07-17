"""Empirically compare coarse ball-pair gradients with fine node-pair gradients."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from deeprobust.graph import utils
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import preprocess

from heir import Heirattack

from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.gradient_metrics import summarize_rows
from gbhaa_experiments.runner import write_json


def _remap_indices(indices: Sequence[int], mapping: Dict[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(index)] for index in indices if int(index) in mapping])


def _sample_induced_graph(
    adj: sp.spmatrix,
    features: sp.spmatrix,
    labels: np.ndarray,
    idx_train: Sequence[int],
    idx_val: Sequence[int],
    idx_test: Sequence[int],
    max_nodes: Optional[int],
    seed: int,
):
    n = adj.shape[0]
    if max_nodes is None or max_nodes >= n:
        return adj, features, labels, np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)
    rng = np.random.RandomState(seed)
    mandatory = []
    for split in (idx_train, idx_val, idx_test):
        if len(split):
            mandatory.append(int(split[0]))
    pool = np.setdiff1d(np.arange(n), np.asarray(mandatory), assume_unique=False)
    take = max(0, max_nodes - len(set(mandatory)))
    selected = np.asarray(sorted(set(mandatory).union(rng.choice(pool, take, replace=False))))
    mapping = {int(old): new for new, old in enumerate(selected)}
    return (
        adj.tocsr()[selected][:, selected],
        features.tocsr()[selected],
        np.asarray(labels)[selected],
        _remap_indices(idx_train, mapping),
        _remap_indices(idx_val, mapping),
        _remap_indices(idx_test, mapping),
    )


def _onehot_labels(labels: torch.Tensor, nclass: int) -> torch.Tensor:
    return torch.zeros((len(labels), nclass), device=labels.device).scatter_(
        1, labels.reshape(-1, 1), 1.0
    )


def _guided_partition(
    attack: Heirattack,
    embeddings: torch.Tensor,
    features: torch.Tensor,
    adj: sp.spmatrix,
    n_clusters: int,
) -> np.ndarray:
    scores = attack.compute_global_ppr(
        fea=features,
        adj=adj,
        topk_ratio=attack.global_important_ratio,
        alpha=attack.global_ppr_alpha,
        N=attack.global_ppr_iters,
        use_cos_sim=True,
        seed_strategy=attack.global_seed_strategy,
    )
    n = adj.shape[0]
    important_k = min(
        n, max(n_clusters, int(round(attack.global_important_ratio * n)))
    )
    targets = np.argsort(-scores)[:important_k]
    clusterer = attack._make_clusterer(n_clusters)
    if len(targets) >= n_clusters:
        clusterer.fit_predict(embeddings[targets])
        assignments = clusterer.fit_predict(
            embeddings, centroids=clusterer.centroids
        )
    else:
        assignments = clusterer.fit_predict(embeddings)
    return assignments.detach().cpu().numpy().astype(np.int64)


def _coarsen(
    adj: sp.spmatrix,
    features: np.ndarray,
    labels_l: np.ndarray,
    labels_u: np.ndarray,
    assignments: np.ndarray,
):
    unique = np.unique(assignments)
    remap = {int(old): new for new, old in enumerate(unique)}
    assignments = np.asarray([remap[int(value)] for value in assignments])
    k = len(unique)
    n = len(assignments)
    membership = sp.csr_matrix(
        (np.ones(n), (np.arange(n), assignments)), shape=(n, k)
    )
    sizes = np.asarray(membership.sum(axis=0)).reshape(-1)
    coarse_adj = (membership.T @ adj.tocsr() @ membership).toarray().astype(np.float32)
    coarse_adj[np.diag_indices(k)] += np.maximum(sizes - 1.0, 0.0)
    coarse_features = np.asarray(membership.T @ features) / sizes[:, None]
    coarse_labels_l = np.asarray(membership.T @ labels_l)
    coarse_labels_u = np.asarray(membership.T @ labels_u)
    return assignments, sizes, coarse_adj, coarse_features, coarse_labels_l, coarse_labels_u


def _pair_reference_rows(
    coarse_grad: np.ndarray,
    fine_grad: np.ndarray,
    adj: sp.spmatrix,
    assignments: np.ndarray,
) -> List[Dict[str, object]]:
    adj = adj.tocsr()
    rows: List[Dict[str, object]] = []
    k = int(assignments.max()) + 1
    for left in range(k):
        left_nodes = np.flatnonzero(assignments == left)
        for right in range(left, k):
            right_nodes = np.flatnonzero(assignments == right)
            if left == right:
                local_u, local_v = np.triu_indices(len(left_nodes), k=1)
                u = left_nodes[local_u]
                v = left_nodes[local_v]
                capacity = len(left_nodes) * (len(left_nodes) - 1) // 2
            else:
                u = np.repeat(left_nodes, len(right_nodes))
                v = np.tile(right_nodes, len(left_nodes))
                capacity = len(left_nodes) * len(right_nodes)
            if len(u) == 0:
                continue
            edge_state = np.asarray(adj[u, v]).reshape(-1) > 0
            # One undirected flip changes both A[u,v] and A[v,u], so score the
            # directional derivative along that symmetric perturbation.
            gradient_values = fine_grad[u, v] + fine_grad[v, u]
            fine_scores = np.where(edge_state, -gradient_values, gradient_values)
            best_index = int(np.argmax(fine_scores))
            best_fine_action = "del" if edge_state[best_index] else "add"
            current_edges = int(edge_state.sum())

            gradient = float(
                2.0 * coarse_grad[left, left]
                if left == right
                else coarse_grad[left, right] + coarse_grad[right, left]
            )
            coarse_candidates: List[Tuple[float, str]] = []
            if current_edges < capacity:
                coarse_candidates.append((gradient, "add"))
            if current_edges > 0:
                coarse_candidates.append((-gradient, "del"))
            if not coarse_candidates:
                continue
            coarse_score, coarse_action = max(coarse_candidates, key=lambda item: item[0])
            rows.append(
                {
                    "pair": f"{left}:{right}",
                    "left": left,
                    "right": right,
                    "left_size": len(left_nodes),
                    "right_size": len(right_nodes),
                    "capacity": capacity,
                    "current_edges": current_edges,
                    "coarse_score": float(coarse_score),
                    "coarse_action": coarse_action,
                    "fine_max_score": float(fine_scores[best_index]),
                    "fine_mean_score": float(fine_scores.mean()),
                    "fine_action": best_fine_action,
                    "fine_u": int(u[best_index]),
                    "fine_v": int(v[best_index]),
                    "action_agreement": coarse_action == best_fine_action,
                }
            )
    return rows


def _write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="citeseer")
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--clusters", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--coarsen-method", choices=["gb", "kmeans"], default="gb")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--miter", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--global-important-ratio", type=float, default=0.10)
    parser.add_argument("--global-ppr-alpha", type=float, default=0.15)
    parser.add_argument("--global-ppr-iters", type=int, default=30)
    parser.add_argument(
        "--global-seed-strategy", choices=["uniform", "degree", "label"], default="degree"
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gbhaa_experiments/results/coarse_gradient"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    data = Dataset(root=str(REPO_ROOT / "Data"), name=args.dataset, setting="nettack", seed=args.seed)
    adj, features, labels, idx_train, idx_val, idx_test = _sample_induced_graph(
        data.adj,
        data.features,
        data.labels,
        data.idx_train,
        data.idx_val,
        data.idx_test,
        args.max_nodes,
        args.seed,
    )
    idx_unlabeled = np.union1d(idx_val, idx_test)
    adj_t, features_t, labels_t = preprocess(
        adj, features, labels, preprocess_adj=False
    )
    features_t = features_t.to(device)
    labels_t = labels_t.to(device)
    surrogate = GCN(
        nfeat=features_t.shape[1],
        nclass=int(labels_t.max().item()) + 1,
        nhid=args.hidden,
        dropout=0.5,
        with_relu=False,
        with_bias=True,
        weight_decay=5e-4,
        device=device,
    ).to(device)
    surrogate.fit(features_t, adj_t, labels_t, idx_train, train_iters=args.epochs)

    attack_args = SimpleNamespace(
        coarsen_method=args.coarsen_method,
        seed=args.seed,
        global_important_ratio=args.global_important_ratio,
        global_ppr_alpha=args.global_ppr_alpha,
        global_ppr_iters=args.global_ppr_iters,
        global_seed_strategy=args.global_seed_strategy,
        freeze_structure_features=True,
        use_text_attack=False,
        dataset=args.dataset,
    )
    attack = Heirattack(
        model=surrogate,
        nnodes=adj.shape[0],
        feature_shape=features_t.shape,
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=1.0,
        train_iters=args.miter,
        levels=2,
        gb_data=data,
        use_oracle=False,
        lr=args.lr,
        args=attack_args,
        features=features_t,
    ).to(device)
    # The canonical attack sets these flags at the start of
    # meta_attack_multi_step(). This diagnostic calls the lower-level gradient
    # routines directly, so it must establish the same input representation.
    attack.sparse_features = False
    attack.sparse_adj = False

    dense_adj = torch.tensor(adj.toarray(), dtype=torch.float32, device=device)
    dense_adj.requires_grad_(True)
    attack.cur_adj = dense_adj
    normalized = utils.normalize_adj_tensor(dense_adj)
    attack.inner_train(features_t, normalized, idx_train, idx_unlabeled, labels_t)
    embeddings = attack.get_embeddings(features_t, normalized)
    labels_st = attack.self_training_label(labels_t, idx_train)
    onehot = _onehot_labels(labels_st, attack.nclass)
    labels_l = torch.zeros_like(onehot)
    labels_u = torch.zeros_like(onehot)
    labels_l[idx_train] = onehot[idx_train]
    labels_u[idx_unlabeled] = onehot[idx_unlabeled]
    fine_grad, _ = attack.get_meta_grad(
        features_t, normalized, labels_l, labels_u
    )
    fine_grad_np = fine_grad.detach().cpu().numpy()

    feature_np = features_t.detach().cpu().numpy()
    labels_l_np = labels_l.detach().cpu().numpy()
    labels_u_np = labels_u.detach().cpu().numpy()
    output_dir = (REPO_ROOT / args.output_dir / args.dataset / f"seed_{args.seed}").resolve()
    summaries = []
    for requested_clusters in args.clusters:
        clusters = min(max(2, requested_clusters), adj.shape[0])
        assignments = _guided_partition(
            attack, embeddings, features_t, adj, clusters
        )
        (
            assignments,
            sizes,
            coarse_adj,
            coarse_features,
            coarse_labels_l,
            coarse_labels_u,
        ) = _coarsen(
            adj,
            feature_np,
            labels_l_np,
            labels_u_np,
            assignments,
        )
        attack.cur_adj = torch.tensor(
            coarse_adj, dtype=torch.float32, device=device, requires_grad=True
        )
        coarse_norm = utils.normalize_adj_tensor(attack.cur_adj)
        coarse_grad, _ = attack.get_meta_grad(
            torch.tensor(coarse_features, dtype=torch.float32, device=device),
            coarse_norm,
            torch.tensor(coarse_labels_l, dtype=torch.float32, device=device),
            torch.tensor(coarse_labels_u, dtype=torch.float32, device=device),
        )
        rows = _pair_reference_rows(
            coarse_grad.detach().cpu().numpy(), fine_grad_np, adj, assignments
        )
        summary = summarize_rows(rows, topk=args.topk)
        summary.update(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "requested_clusters": requested_clusters,
                "actual_clusters": int(assignments.max()) + 1,
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "coarsen_method": args.coarsen_method,
            }
        )
        summaries.append(summary)
        _write_rows(output_dir / f"pairs_k{requested_clusters}.csv", rows)
        write_json(output_dir / f"summary_k{requested_clusters}.json", summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))

    write_json(
        output_dir / "summary.json",
        {"runs": summaries, "reference": "fine node-level first-order gradient"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
