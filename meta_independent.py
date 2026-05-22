"""
Independent structure/attribute attack experiment.

Run this file with the same arguments as meta.py. The structure attack still
selects an edge by the original hierarchical search, while the text/attribute
attack selects its own seed nodes from feature-gradient scores and then expands
them with local AE-PPR. Both perturbations are applied in the same step.
"""

import argparse
import math
import os
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from deeprobust.graph import utils
from tqdm import tqdm

import heir as heir_module


_BASE_HEIR_ATTACK = heir_module.Heirattack
_INDEPENDENT_CONFIG = SimpleNamespace(
    text_seed_count=2,
    text_seed_strategy="feature_grad",
    text_seed_pool="all",
)


class IndependentHeirattack(_BASE_HEIR_ATTACK):
    def __init__(self, *args, **kwargs):
        patched_class = heir_module.Heirattack
        heir_module.Heirattack = _BASE_HEIR_ATTACK
        try:
            super().__init__(*args, **kwargs)
        finally:
            heir_module.Heirattack = patched_class

    def get_feature_attack_scores(self, features, adj_norm, labels_l, labels_u):
        X = features.detach()
        if X.is_sparse:
            X = X.to_dense()
        X = X.clone().requires_grad_(True)

        hidden = X
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)
        attack_loss = torch.sum(-output * labels_l) / (torch.sum(labels_l) + 1e-8)
        attack_loss += torch.sum(-output * labels_u) / (torch.sum(labels_u) + 1e-8)

        feature_grad = torch.autograd.grad(attack_loss, X, retain_graph=False)[0]
        scores = torch.norm(feature_grad, p=2, dim=1)
        return scores.detach().cpu().numpy()

    def select_independent_text_seeds(
        self,
        feature_scores,
        idx_train,
        idx_unlabeled,
    ):
        scores = np.asarray(feature_scores, dtype=np.float64).copy()
        strategy = _INDEPENDENT_CONFIG.text_seed_strategy

        if strategy == "global_ppr" and hasattr(self, "global_ppr_scores"):
            scores = np.asarray(self.global_ppr_scores, dtype=np.float64).copy()
        elif strategy == "hybrid" and hasattr(self, "global_ppr_scores"):
            grad_scores = scores / (np.max(scores) + 1e-12)
            ppr_scores = np.asarray(self.global_ppr_scores, dtype=np.float64)
            ppr_scores = ppr_scores / (np.max(ppr_scores) + 1e-12)
            scores = 0.5 * grad_scores + 0.5 * ppr_scores

        pool = _INDEPENDENT_CONFIG.text_seed_pool
        if pool == "unlabeled":
            mask = np.zeros(self.nnodes, dtype=bool)
            mask[np.asarray(idx_unlabeled, dtype=np.int64)] = True
            scores[~mask] = -np.inf
        elif pool == "train":
            mask = np.zeros(self.nnodes, dtype=bool)
            mask[np.asarray(idx_train, dtype=np.int64)] = True
            scores[~mask] = -np.inf

        attacked_nodes = getattr(self, "_attacked_nodes", {})
        seeds = []
        for node in np.argsort(-scores).tolist():
            if not np.isfinite(scores[node]):
                continue
            if attacked_nodes.get(node, 0) >= self.text_attack_max_visits:
                continue
            seeds.append(int(node))
            if len(seeds) >= _INDEPENDENT_CONFIG.text_seed_count:
                break
        return np.asarray(seeds, dtype=np.int64), scores

    def collect_text_nodes_from_seeds(self, seeds, full_features, full_adj_cpu):
        candidate_scores = {}
        for seed in np.asarray(seeds, dtype=np.int64).tolist():
            topk_nodes, scores = self.ppr_topk_from_seed(
                fea=full_features,
                adj=full_adj_cpu,
                seed_u=int(seed),
                topk_ratio=self.text_topk_ratio,
                alpha=self.text_ppr_alpha,
                N=self.text_ppr_iters,
                use_cos_sim=True,
            )
            for node in topk_nodes.tolist():
                score = float(scores[node])
                if score > candidate_scores.get(int(node), -np.inf):
                    candidate_scores[int(node)] = score

        ordered = sorted(
            candidate_scores.keys(), key=lambda node: candidate_scores[node], reverse=True
        )
        ordered_scores = [candidate_scores[node] for node in ordered]
        return (
            np.asarray(ordered, dtype=np.int64),
            np.asarray(ordered_scores, dtype=np.float32),
        )

    def filter_text_attack_nodes(self, candidate_nodes, candidate_scores):
        if not hasattr(self, "_attacked_nodes"):
            self._attacked_nodes = {}

        candidate_nodes = np.asarray(candidate_nodes, dtype=np.int64)
        candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
        if len(candidate_scores) != len(candidate_nodes):
            candidate_scores = np.zeros(len(candidate_nodes), dtype=np.float32)

        selected = []
        skipped = 0
        for pos in np.argsort(-candidate_scores).tolist():
            node = int(candidate_nodes[pos])
            if self._attacked_nodes.get(node, 0) >= self.text_attack_max_visits:
                skipped += 1
                continue
            selected.append(node)

        per_step = getattr(self.args, "text_attack_nodes", None)
        if per_step is not None and int(per_step) > 0:
            selected = selected[: int(per_step)]

        return np.asarray(selected, dtype=np.int64), skipped

    def meta_attack_multi_step(
        self,
        ori_features,
        ori_adj,
        labels,
        idx_train,
        idx_unlabeled,
        n_perturbations,
        n_step=1,
        ll_constraint=True,
        ll_cutoff=0.004,
        type="Meta-Both",
    ):
        self.sparse_features = sp.issparse(ori_features)
        self.sparse_adj = sp.issparse(ori_adj)

        full_adj, full_features, labels = utils.to_tensor(
            ori_adj, ori_features, labels, device=self.device
        )
        base_structure_features = full_features.detach().clone()

        labels_st = self.self_training_label(labels, idx_train)
        labels_oh_l = torch.zeros(self.nnodes, self.nclass).to(self.device)
        labels_oh_ul = torch.zeros(self.nnodes, self.nclass).to(self.device)
        labels_oh_ul[idx_unlabeled] = labels_oh_ul[idx_unlabeled].scatter_(
            1, labels_st[idx_unlabeled].unsqueeze(1), 1
        )
        labels_oh_l[idx_train] = labels_oh_l[idx_train].scatter_(
            1, labels_st[idx_train].unsqueeze(1), 1
        )

        n_turns = math.ceil(n_perturbations * 1.0 / n_step)
        tot_perturbs = 0
        full_adj_cpu = ori_adj.copy()
        added, deled = sp.csr_matrix(ori_adj.shape), sp.csr_matrix(ori_adj.shape)

        num_add, num_del, depth = 0, 0, 0

        print("\nIndependent attribute selection enabled")
        print(
            "  Attribute seeds: "
            f"strategy={_INDEPENDENT_CONFIG.text_seed_strategy}, "
            f"count={_INDEPENDENT_CONFIG.text_seed_count}, "
            f"pool={_INDEPENDENT_CONFIG.text_seed_pool}"
        )

        for turn in tqdm(range(n_turns), desc="Perturbing graph"):
            search_features = (
                base_structure_features
                if self.freeze_structure_features
                else full_features
            )
            self.full_adj = utils.sparse_mx_to_torch_sparse_tensor(full_adj_cpu).to(
                self.device
            )
            adj_norm = utils.normalize_adj_tensor(self.full_adj, sparse=True)
            self.inner_train(
                search_features, adj_norm, idx_train, idx_unlabeled, labels
            )
            embeddings = self.get_embeddings(search_features, adj_norm)

            self.global_ppr_scores = self.compute_global_ppr(
                fea=search_features,
                adj=full_adj_cpu,
                topk_ratio=self.global_important_ratio,
                alpha=self.global_ppr_alpha,
                N=self.global_ppr_iters,
                use_cos_sim=True,
                seed_strategy=self.global_seed_strategy,
            )
            global_k = min(
                self.nnodes,
                max(self.M, int(round(self.global_important_ratio * self.nnodes))),
            )
            global_important_nodes = np.argsort(-self.global_ppr_scores)[:global_k]
            feature_node_scores = None
            if self.attack_features and self.use_text_attack:
                feature_node_scores = self.get_feature_attack_scores(
                    full_features,
                    adj_norm,
                    labels_oh_l,
                    labels_oh_ul,
                )

            pool = [range(self.nnodes)]
            childs = [[]]
            parents = [-1]
            levels = [1]
            cur = 0
            while cur < len(pool):
                subgraph = pool[cur]
                n = len(subgraph)

                if n <= self.M:
                    if len(subgraph) > 1:
                        for node in subgraph:
                            childs[cur].append(len(pool))
                            pool.append([node])
                            childs.append([])
                            parents.append(cur)
                            levels.append(levels[cur] + 1)
                    cur += 1
                    continue

                subgraph_list = list(subgraph)
                subgraph_set = set(subgraph_list)
                targets = [
                    node for node in global_important_nodes if node in subgraph_set
                ]

                if len(targets) >= self.M:
                    self.gb_cluster.fit_predict(embeddings[targets, :])
                    cid = self.gb_cluster.fit_predict(
                        embeddings[subgraph_list, :],
                        centroids=self.gb_cluster.centroids,
                    )
                else:
                    cid = self.gb_cluster.fit_predict(embeddings[subgraph_list, :])

                for _ in range(self.M):
                    childs[cur].append(len(pool))
                    pool.append([])
                    childs.append([])
                    parents.append(cur)
                    levels.append(levels[cur] + 1)

                for node_pos in range(n):
                    cluster_id = cid[node_pos].item()
                    pool[-cluster_id - 1].append(subgraph[node_pos])

                cur += 1

            self.node2cluster = {}
            for pool_id in range(len(pool)):
                if len(childs[pool_id]) == 0 and len(pool[pool_id]) > 0:
                    for node_idx in pool[pool_id]:
                        self.node2cluster[node_idx] = pool_id

            for _ in range(n_step):
                tot_perturbs += 1
                if tot_perturbs > n_perturbations:
                    break

                while len((full_adj_cpu - ori_adj).nonzero()[0]) < tot_perturbs * 2:
                    inpool_set = set(childs[0])
                    status = "unknown"
                    targetI, targetJ = 0, 0

                    while True:
                        depth += 1

                        inpool = list(inpool_set)
                        n = len(inpool)
                        adj_inpool = np.zeros((n, n))
                        added_inpool = np.zeros((n, n))
                        deled_inpool = np.zeros((n, n))
                        feature_inpool = torch.zeros((n, self.nfeat)).to(self.device)
                        labels_inpool_l = torch.zeros((n, self.nclass)).to(self.device)
                        labels_inpool_ul = torch.zeros((n, self.nclass)).to(self.device)

                        sizes = torch.zeros((n, 1)).to(self.device)
                        cids = np.zeros(self.nnodes)

                        for i in range(n):
                            nodes = pool[inpool[i]]
                            sizes[i] = len(nodes)
                            feature_inpool[i] = torch.mean(
                                search_features[nodes, :], dim=0
                            )
                            if "Both" in type or "Train" in type:
                                labels_inpool_l[i] = torch.sum(
                                    labels_oh_l[nodes, :], dim=0
                                )
                            if "Both" in type or "Self" in type:
                                labels_inpool_ul[i] = torch.sum(
                                    labels_oh_ul[nodes, :], dim=0
                                )
                            for node in nodes:
                                cids[node] = i
                        for row, col in zip(*full_adj_cpu.nonzero()):
                            i, j = int(cids[row]), int(cids[col])
                            adj_inpool[i][j] = adj_inpool[i][j] + 1
                        for row, col in zip(*added.nonzero()):
                            i, j = int(cids[row]), int(cids[col])
                            added_inpool[i][j] = added_inpool[i][j] + 1
                        for row, col in zip(*deled.nonzero()):
                            i, j = int(cids[row]), int(cids[col])
                            deled_inpool[i][j] = deled_inpool[i][j] + 1
                        self.cur_adj = torch.Tensor(adj_inpool).to(self.device)
                        for i in range(n):
                            self.cur_adj[i][i] = self.cur_adj[i][i] + sizes[i] - 1
                        self.cur_adj.requires_grad = True
                        adj_ip_norm = utils.normalize_adj_tensor(self.cur_adj).to(
                            self.device
                        )

                        adj_grad, feature_grad = self.get_meta_grad(
                            feature_inpool,
                            adj_ip_norm,
                            labels_inpool_l,
                            labels_inpool_ul,
                        )

                        posI, posJ = [], []
                        for i in range(n):
                            ip = inpool[i]
                            if parents[ip] == targetI or (
                                ip == targetI and sizes[i] == 1
                            ):
                                posI.append(i)
                        for i in range(n):
                            ip = inpool[i]
                            if parents[ip] == targetJ or (
                                ip == targetJ and sizes[i] == 1
                            ):
                                posJ.append(i)
                        best_score, best_status, I, J = -1e10, "unknown", -1, -1

                        for i in posI:
                            for j in posJ:
                                if self.attack_structure:
                                    if status in ["unknown", "add"]:
                                        possible_edges = (
                                            (sizes[i] - 1) * sizes[i] / 2
                                            if i == j
                                            else sizes[i] * sizes[j]
                                        )
                                        possible_edges = (
                                            possible_edges - deled_inpool[i][j]
                                        )
                                        if possible_edges > adj_inpool[i][j]:
                                            score = adj_grad[i][j]
                                            if score > best_score:
                                                best_score, best_status, I, J = (
                                                    score,
                                                    "add",
                                                    i,
                                                    j,
                                                )
                                    if status in ["unknown", "del"]:
                                        if adj_inpool[i][j] > added_inpool[i][j]:
                                            score = -adj_grad[i][j]
                                            if score > best_score:
                                                best_score, best_status, I, J = (
                                                    score,
                                                    "del",
                                                    i,
                                                    j,
                                                )

                        status, targetI, targetJ = best_status, inpool[I], inpool[J]
                        if sizes[I] == 1 and sizes[J] == 1:
                            break
                        if sizes[I] > 1:
                            inpool_set.remove(inpool[I])
                            for child in childs[inpool[I]]:
                                if len(pool[child]) > 0:
                                    inpool_set.add(child)
                        if sizes[J] > 1 and not I == J:
                            inpool_set.remove(inpool[J])
                            for child in childs[inpool[J]]:
                                if len(pool[child]) > 0:
                                    inpool_set.add(child)

                    row_idx, col_idx = pool[targetI][0], pool[targetJ][0]

                    if (
                        self.attack_features
                        and self.use_text_attack
                        and self.text_generator is not None
                    ):
                        attr_seeds, _ = self.select_independent_text_seeds(
                            feature_node_scores,
                            idx_train,
                            idx_unlabeled,
                        )
                        U, U_scores = self.collect_text_nodes_from_seeds(
                            attr_seeds,
                            full_features,
                            full_adj_cpu,
                        )
                        new_nodes, skipped = self.filter_text_attack_nodes(U, U_scores)

                        if len(new_nodes) > 0:
                            print(
                                f"\n[Step {tot_perturbs}/{n_perturbations}] "
                                "Independent text attack: "
                                f"structure_edge=({row_idx}, {col_idx}), "
                                f"attr_seeds={attr_seeds.tolist()}, "
                                f"selected={len(new_nodes)}/{len(U)}, "
                                f"exhausted={skipped}"
                            )

                            text_embeddings = self.get_embeddings(
                                full_features, adj_norm
                            )
                            new_nodes_tensor = (
                                torch.from_numpy(new_nodes)
                                .long()
                                .to(text_embeddings.device)
                            )
                            round_text_embeddings = text_embeddings[new_nodes_tensor]
                            text_clusters = self._build_round_text_clusters(
                                new_nodes,
                                round_text_embeddings,
                                min_cluster_size=self.text_min_cluster_size,
                                max_cluster_size=self.text_max_cluster_size,
                            )

                            full_features = self.attack_features_with_text(
                                target_nodes=new_nodes,
                                full_features=full_features,
                                labels_st=labels_st,
                                budget_per_node=self.text_budget_per_node,
                                target_embeddings=round_text_embeddings,
                                text_clusters=text_clusters,
                            )

                            for node in new_nodes.tolist():
                                self._attacked_nodes[node] = (
                                    self._attacked_nodes.get(node, 0) + 1
                                )
                        else:
                            print(
                                f"\n[Step {tot_perturbs}/{n_perturbations}] "
                                "Independent text attack skipped: no eligible nodes"
                            )

                    full_adj_cpu[row_idx, col_idx] = 1 - full_adj_cpu[row_idx, col_idx]
                    full_adj_cpu[col_idx, row_idx] = 1 - full_adj_cpu[col_idx, row_idx]

                    if status == "add":
                        num_add += 1
                        added[row_idx, col_idx] = 1
                        added[col_idx, row_idx] = 1
                    else:
                        num_del += 1
                        deled[row_idx, col_idx] = 1
                        deled[col_idx, row_idx] = 1

        if self.attack_structure:
            self.modified_adj = full_adj_cpu
        self.modified_features = full_features.detach()

        if self.attack_features:
            print(
                "\nIndependent attack completed: "
                f"Structure perturbations={num_add + num_del}, "
                "attribute perturbations selected independently"
            )
        else:
            print(
                "\nIndependent attack completed: "
                f"Structure perturbations={num_add + num_del} "
                "(feature attack disabled)"
            )


def _consume_independent_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--independent_text_seed_count",
        type=int,
        default=_INDEPENDENT_CONFIG.text_seed_count,
    )
    parser.add_argument(
        "--independent_text_seed_strategy",
        type=str,
        default=_INDEPENDENT_CONFIG.text_seed_strategy,
        choices=["feature_grad", "global_ppr", "hybrid"],
    )
    parser.add_argument(
        "--independent_text_seed_pool",
        type=str,
        default=_INDEPENDENT_CONFIG.text_seed_pool,
        choices=["all", "unlabeled", "train"],
    )
    independent_args, remaining = parser.parse_known_args(argv[1:])

    _INDEPENDENT_CONFIG.text_seed_count = max(
        1, int(independent_args.independent_text_seed_count)
    )
    _INDEPENDENT_CONFIG.text_seed_strategy = (
        independent_args.independent_text_seed_strategy
    )
    _INDEPENDENT_CONFIG.text_seed_pool = independent_args.independent_text_seed_pool

    sys.argv = [argv[0]] + remaining


def main():
    _consume_independent_args(sys.argv)
    heir_module.Heirattack = IndependentHeirattack

    meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta.py")
    runpy.run_path(meta_path, run_name="__main__")


if __name__ == "__main__":
    main()
