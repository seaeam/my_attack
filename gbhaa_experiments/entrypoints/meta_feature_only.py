"""True feature-only control: update attributes while keeping adjacency fixed."""

from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph import utils
from tqdm import tqdm

import heir as heir_module
import meta_independent


REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_ATTACK = meta_independent.IndependentHeirattack


class FeatureOnlyAttack(_BASE_ATTACK):
    """Use independent feature gradients without searching or flipping an edge."""

    def __init__(self, *args, **kwargs):
        patched_class = heir_module.Heirattack
        heir_module.Heirattack = _BASE_ATTACK
        try:
            super().__init__(*args, **kwargs)
        finally:
            heir_module.Heirattack = patched_class

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
        del ll_constraint, ll_cutoff
        self.sparse_features = sp.issparse(ori_features)
        self.sparse_adj = sp.issparse(ori_adj)
        full_adj, full_features, labels = utils.to_tensor(
            ori_adj, ori_features, labels, device=self.device
        )
        del full_adj
        original_features = full_features.detach().clone()
        fixed_adj_cpu = ori_adj.copy()

        labels_st = self.self_training_label(labels, idx_train)
        labels_l = torch.zeros(self.nnodes, self.nclass, device=self.device)
        labels_u = torch.zeros(self.nnodes, self.nclass, device=self.device)
        if "Both" in type or "Self" in type:
            labels_u[idx_unlabeled] = labels_u[idx_unlabeled].scatter_(
                1, labels_st[idx_unlabeled].unsqueeze(1), 1
            )
        if "Both" in type or "Train" in type:
            labels_l[idx_train] = labels_l[idx_train].scatter_(
                1, labels_st[idx_train].unsqueeze(1), 1
            )

        rounds = int(math.ceil(float(n_perturbations) / max(int(n_step), 1)))
        completed = 0
        for _ in tqdm(range(rounds), desc="Feature-only attack"):
            for _ in range(max(int(n_step), 1)):
                if completed >= n_perturbations:
                    break
                completed += 1
                search_features = (
                    original_features
                    if self.freeze_structure_features
                    else full_features
                )
                self.full_adj = utils.sparse_mx_to_torch_sparse_tensor(
                    fixed_adj_cpu
                ).to(self.device)
                adj_norm = utils.normalize_adj_tensor(self.full_adj, sparse=True)
                self.inner_train(
                    search_features, adj_norm, idx_train, idx_unlabeled, labels
                )
                self.global_ppr_scores = self.compute_global_ppr(
                    fea=search_features,
                    adj=fixed_adj_cpu,
                    topk_ratio=self.global_important_ratio,
                    alpha=self.global_ppr_alpha,
                    N=self.global_ppr_iters,
                    use_cos_sim=True,
                    seed_strategy=self.global_seed_strategy,
                )
                if not (
                    self.attack_features
                    and self.use_text_attack
                    and self.text_generator is not None
                ):
                    continue

                feature_scores = self.get_feature_attack_scores(
                    full_features, adj_norm, labels_l, labels_u
                )
                seeds, _ = self.select_independent_text_seeds(
                    feature_scores, idx_train, idx_unlabeled
                )
                candidates, candidate_scores = self.collect_text_nodes_from_seeds(
                    seeds, full_features, fixed_adj_cpu
                )
                selected, skipped = self.filter_text_attack_nodes(
                    candidates, candidate_scores
                )
                if len(selected) == 0:
                    print(
                        f"\n[Feature step {completed}/{n_perturbations}] "
                        "skipped: no eligible nodes"
                    )
                    continue

                embeddings = self.get_embeddings(full_features, adj_norm)
                selected_tensor = torch.from_numpy(selected).long().to(self.device)
                selected_embeddings = embeddings[selected_tensor]
                clusters = self._build_round_text_clusters(
                    selected,
                    selected_embeddings,
                    min_cluster_size=self.text_min_cluster_size,
                    max_cluster_size=self.text_max_cluster_size,
                )
                print(
                    f"\n[Feature step {completed}/{n_perturbations}] "
                    f"seeds={seeds.tolist()}, selected={len(selected)}/{len(candidates)}, "
                    f"exhausted={skipped}"
                )
                full_features = self.attack_features_with_text(
                    target_nodes=selected,
                    full_features=full_features,
                    labels_st=labels_st,
                    budget_per_node=self.text_budget_per_node,
                    target_embeddings=selected_embeddings,
                    text_clusters=clusters,
                )
                for node in selected.tolist():
                    self._attacked_nodes[node] = self._attacked_nodes.get(node, 0) + 1

        self.modified_adj = fixed_adj_cpu
        self.modified_features = full_features.detach()
        print(
            "\nFeature-only attack completed: Structure perturbations=0, "
            f"attribute rounds={completed}"
        )


def main() -> None:
    meta_independent._consume_independent_args(sys.argv)
    heir_module.Heirattack = FeatureOnlyAttack
    runpy.run_path(str(REPO_ROOT / "meta.py"), run_name="__main__")


if __name__ == "__main__":
    main()
