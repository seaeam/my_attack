"""Opt-in saved-action replay; the original attack loop/order is unchanged."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph import utils

import heir as heir_module
from heir import Heirattack as BaseAttack
from gbhaa_experiments.entrypoints import meta_followup as followup_module
from gbhaa_experiments.entrypoints.meta_followup import FollowupAttack
from gbhaa_experiments.action_state_protocol import (
    PROTOCOL, ARMS, actions_hash, independent_restarts, flip_action, rank_on_graph,
    validate_trajectory, protocol_settings, operator_settings, feature_measures,
)
from gbhaa_experiments.action_state_cache import RequestCache
from gbhaa_experiments.followup_protocol import (
    array_hash, as_array, graph_hash, reset_rng, seed_stream, stable_order,
)
from gbhaa_experiments.runner import write_json

ROOT = Path(__file__).resolve().parents[2]


class StructureReady(Exception):
    """Stop the bootstrap before victim evaluation for the shared planner."""


class ActionStateAttack(FollowupAttack):
    @property
    def protocol(self):
        return PROTOCOL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_cache = RequestCache(self.options.cache)
        if self.text_generator is not None:
            if self.text_generator.max_tokens != operator_settings()["generation_max_tokens"]:
                raise ValueError("Action-state protocol requires generation max_tokens=50")
            self.request_cache.attach(self.text_generator)

    def input_metadata(self, features, adj, labels, idx_train, idx_eval, target):
        return {
            "protocol": PROTOCOL, "dataset": self.args.dataset, "seed": self.args.seed,
            "budget": float(self.args.ptb_rate), "variant": self.options.phase,
            "settings": protocol_settings(), "operator_settings": operator_settings(),
            "original_adj_hash": graph_hash(adj), "original_features_hash": array_hash(features),
            "training_nodes_hash": array_hash(np.asarray(idx_train)),
            "target_nodes_hash": array_hash(np.asarray(idx_eval)),
            "evaluation_labels_hash": array_hash(as_array(labels)[np.asarray(idx_eval)]),
            "target_edges": int(target),
        }

    def rank_global_candidates(self, scores, k):
        return stable_order(scores)[:k]

    def record_structural_action(self, adj, row_idx, col_idx):
        action = [int(row_idx), int(col_idx), int(1 - adj[row_idx, col_idx])]
        post = flip_action(adj, action)
        self.actions.append(action)
        self.diagnostics[-1].update(action=action, pre_graph_hash=graph_hash(adj),
            post_graph_hash=graph_hash(post),
            structural_seconds=time.perf_counter() - self._structure_started)

    def finish_text_attack_round(self, stats):
        # Successful unchanged writes are valid treatment outcomes, not failures.
        self.last_text_attack_stats = dict(stats)
        self.diagnostics[-1]["attribute_counts"] = dict(stats)
        for key, value in stats.items():
            self.attribute_counts[key] = self.attribute_counts.get(key, 0) + value
        if (stats["targets"] != 4 or stats["feature_writes"] != 4
                or any(stats[k] for k in ("template_failures", "node_failures", "cluster_failures"))):
            raise ValueError("Action-state round requires four successful writes and zero generation failures")

    def attack_features_with_text(self, *args, **kwargs):
        # Use the original generation/projection implementation with our completion
        # hook, bypassing the older follow-up wrapper's separate counter accumulation.
        return BaseAttack.attack_features_with_text(self, *args, **kwargs)

    def _check_options(self, target):
        for key, value in protocol_settings().items():
            if getattr(self.args, key, None) != value:
                raise ValueError(f"Action-state protocol requires {key}={value!r}")
        if (not self.freeze_structure_features or self.use_oracle
                or self.text_attack_total_nodes is not None or target < 1):
            raise ValueError("Requires frozen features, no oracle/distinct-node cap, and positive edge budget")
        if self.nnodes < 4 or 4 * target > 10 * self.nnodes:
            raise ValueError("Graph/edge budget cannot support four nodes per round with ten visits per node")

    def _save_plan(self, features, adj, labels, idx_train, idx_eval, target, kwargs):
        self.attack_features = False
        self.use_text_attack = False
        started = time.perf_counter()
        BaseAttack.meta_attack_multi_step(self, features, adj, labels,
                                         idx_train, idx_eval, target, **kwargs)
        validate_trajectory(adj, self.modified_adj, self.actions, int(target))
        self.metadata.update(actions=self.actions, actions_hash=actions_hash(self.actions),
            completed_edges=int(target), budget_complete=True,
            modified_adj_hash=graph_hash(self.modified_adj),
            structural_seconds=time.perf_counter()-started)
        sp.save_npz(self.trace_dir / "modified_adj.npz", self.modified_adj)
        write_json(self.trace_dir / "rounds.json", {"rounds": self.diagnostics})
        write_json(self.trace_dir / "trajectory.json", self.metadata)
        raise StructureReady()

    def _load_plan(self):
        trajectory = json.loads(self.options.trajectory.read_text())
        for key in ("protocol", "dataset", "seed", "budget", "settings", "operator_settings",
                    "original_adj_hash", "original_features_hash", "training_nodes_hash",
                    "target_nodes_hash", "evaluation_labels_hash", "target_edges"):
            if trajectory.get(key) != self.metadata[key]:
                raise ValueError(f"Shared trajectory {key} differs from current inputs/configuration")
        if trajectory.get("variant") != "structure" or actions_hash(trajectory["actions"]) != trajectory["actions_hash"]:
            raise ValueError("Invalid saved structural trajectory")
        return trajectory

    def _select_round(self, features, pre, post, action):
        endpoints = action[:2]
        pre_order, pre_scores = rank_on_graph(self, features, pre, endpoints)
        # Execute the edge before the Post diffusion and before all generation.
        self.full_adj = utils.sparse_mx_to_torch_sparse_tensor(post).to(self.device)
        post_order, post_scores = rank_on_graph(self, features, post, endpoints)
        pre_nodes = self.filter_text_attack_nodes(pre_order)[:4]
        post_nodes = self.filter_text_attack_nodes(post_order)[:4]
        phase = self.options.phase
        restarts = endpoints
        if phase == "parallel":
            restarts = independent_restarts(self.nnodes, self.args.seed, self.round_index)
            order, _ = rank_on_graph(self, features, pre, restarts)
        else:
            order = post_order if phase == "endpoint_post" else pre_order
        eligible = self.filter_text_attack_nodes(order)
        nodes = eligible[:4]
        if len(nodes) != 4:
            raise ValueError("Cannot fill the matched four-node round budget")
        union = set(pre_nodes) | set(post_nodes)
        vectors = as_array(features)[endpoints]
        norms = np.linalg.norm(vectors, axis=1)
        similarity = float(np.dot(*vectors) / np.prod(norms)) if np.prod(norms) > 1e-12 else 0.
        graph = post if phase == "endpoint_post" else pre
        self.diagnostics[-1].update(
            pre_graph_hash=graph_hash(pre), post_graph_hash=graph_hash(post),
            diffusion_graph_hash=graph_hash(graph), local_graph="post" if phase == "endpoint_post" else "pre",
            local_edge_value=int(graph[action[0],action[1]]), restart_nodes=list(restarts),
            selected_nodes=nodes.tolist(), candidate_universe=self.nnodes, eligible_count=len(eligible),
            current_features_hash=array_hash(features), probe_pre_nodes=pre_nodes.tolist(),
            probe_post_nodes=post_nodes.tolist(), probe_jaccard=len(set(pre_nodes)&set(post_nodes))/len(union),
            probe_score_l1=float(np.abs(pre_scores-post_scores).sum()),
            action_similarity=float(np.clip(similarity,0,1)), grouping_graph="post",
            grouping_graph_hash=graph_hash(post),
        )
        if phase == "parallel":
            self.diagnostics[-1]["independent_restart_overlap"] = len(set(restarts)&set(endpoints))
        return nodes

    def _replay(self, features, adj, labels, idx_train, idx_eval, target):
        trajectory = self._load_plan()
        state = adj.copy()
        planned_final = state
        for action in trajectory["actions"]:
            planned_final = flip_action(planned_final, action)
        validate_trajectory(adj, planned_final, trajectory["actions"], int(target))
        self.sparse_features=False; self.sparse_adj=True
        _, current, labels_t = utils.to_tensor(adj, features, labels, device=self.device)
        original_t = current.detach().clone()
        labels_st = self.self_training_label(labels_t, idx_train)
        self._attacked_nodes = {}
        updates = []; update_nodes = []
        self.metadata.update(trajectory_hash=trajectory["actions_hash"], selector="ae_pr",
            local_graph="post" if self.options.phase=="endpoint_post" else "pre",
            global_source="degree", generation_seed=seed_stream(self.args.seed,"generation"),
            generation_temperature=0., generation_model=self.text_generator.model_name,
            vocabulary_fallback=getattr(self.text_generator,"uses_fallback_vocabulary",None),
            vocabulary_partial=getattr(self.text_generator,"uses_partial_vocabulary",None))
        for index, action in enumerate(trajectory["actions"]):
            self.round_index=index
            self.diagnostics.append({"round":index,"action":action})
            pre=state
            post=flip_action(pre,action)
            # Same trained grouping model across arms: frozen original features,
            # shared pre-graph, fresh structural random stream each round.
            reset_rng(seed_stream(self.args.seed,"structural",index))
            pre_t=utils.sparse_mx_to_torch_sparse_tensor(pre).to(self.device)
            pre_norm=utils.normalize_adj_tensor(pre_t,sparse=True)
            BaseAttack.inner_train(self,original_t,pre_norm,idx_train,idx_eval,labels_t)
            nodes=self._select_round(current,pre,post,action)
            state=post
            self.actions.append(action)
            post_norm=utils.normalize_adj_tensor(self.full_adj,sparse=True)
            with torch.no_grad():
                embeddings=self.get_embeddings(current,post_norm)[torch.as_tensor(nodes,device=self.device)]
            reset_rng(seed_stream(self.args.seed,"attribute",index))
            clusters=self._build_round_text_clusters(nodes,embeddings,
                min_cluster_size=self.text_min_cluster_size,max_cluster_size=self.text_max_cluster_size)
            self.template_cache={}  # The persistent cache uses the complete actual request.
            current=self.attack_features_with_text(target_nodes=nodes,full_features=current,
                labels_st=labels_st,budget_per_node=self.text_budget_per_node,
                target_embeddings=embeddings,text_clusters=clusters)
            for node in nodes:
                self._attacked_nodes[int(node)]=self._attacked_nodes.get(int(node),0)+1
            update_nodes.append(nodes.copy());updates.append(as_array(current)[nodes].copy())
            print(f"Action-state {self.options.phase}: round {index+1}/{target}, nodes={nodes.tolist()}",flush=True)
        self.modified_adj=state;self.modified_features=current.detach()
        validate_trajectory(adj,state,self.actions,int(target))
        self.metadata.update(actions=self.actions,actions_hash=actions_hash(self.actions),
            modified_adj_hash=graph_hash(state),completed_edges=int(target),budget_complete=True,
            structural_seconds=trajectory.get("structural_seconds",0.),
            attribute_node_visits=sum(self._attacked_nodes.values()),
            visited_attribute_nodes=len(self._attacked_nodes),attribute_counts=self.attribute_counts,
            api_counts=getattr(self.text_generator,"api_stats",{}),request_cache=self.request_cache.stats,
            generation_request_keys=sorted(self.request_cache.used_keys),
            modified_features_hash=array_hash(self.modified_features),
            **feature_measures(features,self.modified_features))
        if self.metadata["attribute_node_visits"] != 4*target or self.attribute_counts.get("feature_writes") != 4*target:
            raise ValueError("Incomplete matched attribute update budget")
        sp.save_npz(self.trace_dir/"modified_adj.npz",state)
        np.savez_compressed(self.trace_dir/"features.npz",original=features,modified=as_array(current))
        np.savez_compressed(self.trace_dir/"feature_updates.npz",nodes=np.stack(update_nodes),values=np.stack(updates))
        write_json(self.trace_dir/"rounds.json",{"rounds":self.diagnostics})
        write_json(self.trace_dir/"attack.json",self.metadata)

    def meta_attack_multi_step(self, ori_features, ori_adj, labels, idx_train,
                               idx_unlabeled, n_perturbations, **kwargs):
        if kwargs.get("n_step",1)!=1:
            raise ValueError("Action-state protocol requires one edge per round")
        self._check_options(n_perturbations)
        features=ori_features.toarray() if sp.issparse(ori_features) else as_array(ori_features)
        features=np.asarray(features,dtype=np.float32)
        adj=sp.csr_matrix(ori_adj)
        self.trace_dir.mkdir(parents=True,exist_ok=True)
        self.metadata=self.input_metadata(features,adj,labels,idx_train,idx_unlabeled,n_perturbations)
        sp.save_npz(self.trace_dir/"original_adj.npz",adj)
        try:
            if self.options.phase=="structure":
                self._save_plan(features,adj,labels,idx_train,idx_unlabeled,n_perturbations,kwargs)
            else:
                self._replay(features,adj,labels,idx_train,idx_unlabeled,n_perturbations)
        except StructureReady:
            raise
        except BaseException:
            write_json(self.trace_dir/"failed_attack.json",{**self.metadata,"budget_complete":False,
                "actions":self.actions,"rounds":self.diagnostics,"attribute_counts":self.attribute_counts})
            raise


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action-state-phase",dest="phase",choices=("structure",*ARMS),required=True)
    parser.add_argument("--action-state-artifacts",dest="artifacts",type=Path,required=True)
    parser.add_argument("--action-state-trajectory",dest="trajectory",type=Path,required=True)
    parser.add_argument("--action-state-cache",dest="cache",type=Path,required=True)
    options,remaining=parser.parse_known_args()
    options.protocol=PROTOCOL;options.selector="ae_pr";options.global_transition="attribute"
    options.local_graph="post" if options.phase=="endpoint_post" else "pre"
    followup_module._OPTIONS=options
    if options.phase!="structure":
        api_key=os.environ.get("OPENAI_API_KEY")
        if not api_key:raise SystemExit("OPENAI_API_KEY is required (local Ollama: ollama)")
        remaining += ["--openai_api_key",api_key]
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1);reset_rng(0)
    sys.argv=[str(ROOT/"meta.py"),*remaining]
    heir_module.Heirattack=ActionStateAttack
    try:
        runpy.run_path(str(ROOT/"meta.py"),run_name="__main__")
    except StructureReady:
        print("Shared structural trajectory complete; no generation/victim evaluation run.",flush=True)
    finally:
        heir_module.Heirattack=BaseAttack


if __name__=="__main__":main()
