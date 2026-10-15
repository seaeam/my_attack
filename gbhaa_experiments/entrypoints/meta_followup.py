"""Opt-in follow-up attacks. Importing this module never starts an experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

import heir as heir_module
from gbhaa_experiments.followup_protocol import (
    PROTOCOL, array_hash, as_array, graph_hash, prediction_metrics,
    reset_rng, seed_stream, selector_order, stable_order, configure_generation,
)
from gbhaa_experiments.runner import write_json
from gbhaa_experiments.followup_config import (
    SELECTOR_PROTOCOL, FACTORIAL_PROTOCOL, pr_solver_settings, factorial_operator_settings,
)

_BASE = heir_module.Heirattack
_OPTIONS = None
ROOT = Path(__file__).resolve().parents[2]


def validate_protocol_options(options):
    protocol = getattr(options, "protocol", PROTOCOL)
    if protocol == SELECTOR_PROTOCOL:
        if options.local_graph != "pre" or options.selector not in ("ae_pr", "pr", "degree", "random"):
            raise ValueError("PR selector protocol requires pre and AE-PR/PR/Degree/Random")
    elif protocol == FACTORIAL_PROTOCOL:
        if (options.local_graph != "pre" or options.selector not in ("ppr", "ae_pr")
                or options.global_transition not in ("topology", "attribute")):
            raise ValueError("Factorial protocol requires pre, endpoint PPR/AE-PR and topology/attribute weights")
    elif protocol != PROTOCOL or options.selector == "pr":
        raise ValueError("Ordinary PR requires the versioned PR selector protocol")


class FollowupAttack(_BASE):
    @property
    def protocol(self):
        return getattr(self.options, "protocol", PROTOCOL)

    def __init__(self, *args, **kwargs):
        # Check parsed state, not option spelling (argparse accepts abbreviations).
        arguments = kwargs.get("args") if "args" in kwargs else (args[2] if len(args) > 2 else None)
        if kwargs.get("use_oracle", False) or getattr(arguments, "oracle", False):
            raise ValueError("Oracle labels are prohibited in the follow-up protocol")
        # The legacy base uses super(Heirattack, self), so patch only during init.
        patched = heir_module.Heirattack
        heir_module.Heirattack = _BASE
        try:
            super().__init__(*args, **kwargs)
        finally:
            heir_module.Heirattack = patched
        self.options = _OPTIONS
        self.followup_edge_first = self.options.local_graph == "post"
        self.round_index = -1
        self.actions = []
        self.diagnostics = []
        self.evaluations = {}
        self.attribute_counts = {}
        self.trace_dir = self.options.artifacts
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if not self.freeze_structure_features:
            raise ValueError("Follow-up requires frozen structural features")
        if self.text_generator is not None:
            if self.text_generator.llm_type != "gpt":
                raise ValueError("Follow-up requires a seed-capable OpenAI-compatible backend")
            configure_generation(self.text_generator, seed_stream(self.args.seed, "generation"))

    def inner_train(self, *args, **kwargs):
        self.round_index += 1
        self._structure_started = time.perf_counter()
        reset_rng(seed_stream(self.args.seed, "structural", self.round_index))
        return super().inner_train(*args, **kwargs)

    def rank_global_candidates(self, scores, k):
        if self.protocol != FACTORIAL_PROTOCOL:
            # The previous loop used argsort directly, including for uniform.
            return super().rank_global_candidates(scores, k)
        selected = stable_order(scores)[:k]
        self.diagnostics[-1].update({
            "global_selected_nodes": selected.tolist(),
            "global_unique_scores": int(np.unique(scores).size),
            "global_cutoff_tie_count": int(np.count_nonzero(scores == scores[selected[-1]])) if len(selected) else 0,
        })
        return selected

    def attack_features_with_text(self, *args, **kwargs):
        self.last_text_attack_stats = {}
        try:
            result = super().attack_features_with_text(*args, **kwargs)
            if any(self.last_text_attack_stats.get(k, 0) for k in
                   ("template_failures", "node_failures", "cluster_failures")):
                raise RuntimeError("Partial attribute-generation failure; excluded from strict matched comparison")
            return result
        finally:
            for key, count in self.last_text_attack_stats.items():
                self.attribute_counts[key] = self.attribute_counts.get(key, 0) + count
            self.diagnostics[-1]["attribute_counts"] = dict(self.last_text_attack_stats)

    def compute_global_ppr(self, *args, **kwargs):
        kwargs["use_cos_sim"] = self.options.global_transition == "attribute"
        scores = super().compute_global_ppr(*args, **kwargs)
        self.diagnostics.append({"round": self.round_index,
                                 "global_min": float(scores.min()),
                                 "global_max": float(scores.max()),
                                 "global_range": float(np.ptp(scores))})
        return scores

    def record_structural_action(self, adj, row_idx, col_idx):
        # Observe the current pre-flip loop without changing its graph or RNG.
        u, v = int(row_idx), int(col_idx)
        if u == v or adj[u, v] not in (0, 1) or adj[u, v] != adj[v, u]:
            raise ValueError("Expected a binary undirected non-self structural action")
        self.actions.append([u, v, int(1 - adj[u, v])])
        self.diagnostics[-1].update({"action": self.actions[-1],
            "local_graph": "pre", "local_edge_value": int(adj[u, v]),
            "structural_seconds": time.perf_counter() - self._structure_started})

    def select_text_candidate_nodes(self, fea, adj, row_idx, col_idx):
        order, scores = selector_order(
            self, fea, adj, [row_idx, col_idx], self.options.selector,
            self.text_ppr_alpha, self.text_ppr_iters,
            seed_stream(self.args.seed, "selector", self.round_index))
        # Common feasibility rules, then the same per-round cap. Legacy runs
        # instead truncate first; this intentional protocol change is versioned.
        eligible = self.filter_text_attack_nodes(order)
        selected = eligible[:self._text_candidate_budget(adj.shape[0])]
        self.diagnostics[-1].update({"candidate_universe": int(adj.shape[0]),
                                    "eligible_count": len(eligible),
                                    "selected_nodes": selected.tolist()})
        return selected

    def meta_attack_multi_step(self, ori_features, ori_adj, labels, idx_train,
                               idx_unlabeled, n_perturbations, *args, **kwargs):
        if kwargs.get("n_step", 1) != 1:
            raise ValueError("Follow-up protocol requires one edge per round")
        operator_settings = None
        if self.protocol == FACTORIAL_PROTOCOL:
            validate_protocol_options(self.options)
            operator_settings = factorial_operator_settings(self.global_ppr_alpha,
                self.global_ppr_iters, self.text_ppr_alpha, self.text_ppr_iters)
            if (self.global_seed_strategy != "degree" or not self.freeze_structure_features
                    or operator_settings != factorial_operator_settings()):
                raise ValueError("Factorial protocol requires degree source, frozen structural features and fixed alpha/iterations")
        self.metadata = {"protocol": self.protocol, "seed": self.args.seed,
            "dataset": self.args.dataset, "target_edges": int(n_perturbations),
            "original_adj_hash": graph_hash(ori_adj),
            "original_features_hash": array_hash(ori_features),
            "training_nodes_hash": array_hash(np.asarray(idx_train)),
            "target_nodes_hash": array_hash(np.asarray(idx_unlabeled)),
            "evaluation_labels_hash": array_hash(as_array(labels)[np.asarray(idx_unlabeled)]),
            "selector": self.options.selector, "local_graph": "pre",
            "requested_local_graph": self.options.local_graph,
            "global_transition": self.options.global_transition,
            "global_source": self.global_seed_strategy,
            "generation_seed": seed_stream(self.args.seed, "generation"),
            "generation_temperature": 0.0,
            "vocabulary_fallback": getattr(self.text_generator, "uses_fallback_vocabulary", None),
            "vocabulary_partial": getattr(self.text_generator, "uses_partial_vocabulary", None),
            "generation_model": getattr(self.text_generator, "model_name", None)}
        if self.options.selector == "pr":
            self.metadata["pr_solver"] = pr_solver_settings(self.text_ppr_alpha)
        if operator_settings is not None:
            self.metadata["operator_settings"] = operator_settings
        sp.save_npz(self.trace_dir / "original_adj.npz", sp.csr_matrix(ori_adj))
        try:
            result = super().meta_attack_multi_step(ori_features, ori_adj, labels,
                        idx_train, idx_unlabeled, n_perturbations, *args, **kwargs)
        except BaseException:
            write_json(self.trace_dir / "failed_attack.json", {**self.metadata,
                "edge_budget_complete": False, "actions": self.actions,
                "attribute_counts": self.attribute_counts,
                "api_counts": getattr(self.text_generator, "api_stats", {}),
                "rounds": self.diagnostics})
            raise
        changed = as_array(self.modified_features) != as_array(ori_features)
        completed = int((self.modified_adj != ori_adj).nnz // 2)
        visits = getattr(self, "_attacked_nodes", {})
        self.metadata.update({"completed_edges": completed,
            "structural_seconds": sum(d["structural_seconds"] for d in self.diagnostics),
            "budget_complete": completed == n_perturbations == len(self.actions),
            "edge_budget_complete": completed == n_perturbations == len(self.actions),
            "attribute_counts": self.attribute_counts,
            "api_counts": getattr(self.text_generator, "api_stats", {}),
            "changed_feature_entries": int(changed.sum()),
            "changed_feature_nodes": int(changed.any(axis=1).sum()),
            "attribute_node_visits": int(sum(visits.values())),
            "visited_attribute_nodes": len(visits),
            "actions_hash": hashlib.sha256(json.dumps(self.actions).encode()).hexdigest(),
            "modified_adj_hash": graph_hash(self.modified_adj)})
        write_json(self.trace_dir / "attack.json", {**self.metadata, "actions": self.actions})
        sp.save_npz(self.trace_dir / "modified_adj.npz", sp.csr_matrix(self.modified_adj))
        write_json(self.trace_dir / "rounds.json", {"rounds": self.diagnostics})
        if not self.metadata["budget_complete"]:
            raise RuntimeError("Incomplete edge budget; run cannot enter matched summaries")
        return result

    def before_victim_evaluation(self, description):
        # Identical initialization/dropout stream across clean/attacked graphs
        # and selector arms. Evaluation never consumes the attack RNG stream.
        reset_rng(seed_stream(self.args.seed, "victim"))

    def record_victim_evaluation(self, description, output, labels, idx_eval):
        key = {"Clean graph": "clean", "Edge-only attack": "edge",
               "Feature-only attack": "feature", "Combined attack": "combined"}[description]
        ids = np.asarray(idx_eval)
        logits = as_array(output)[ids]
        y = as_array(labels)[ids]
        predictions = logits.argmax(axis=1)
        self.evaluations[key] = predictions
        np.savez_compressed(self.trace_dir / f"{key}_predictions.npz",
            node_ids=ids, labels=y, predictions=predictions, log_probabilities=logits)
        if key == "combined":
            metrics = prediction_metrics(self.evaluations["clean"], predictions, y)
            for stage, pred in self.evaluations.items():
                metrics[f"{stage}_accuracy"] = float((pred == y).mean())
            write_json(self.trace_dir / "evaluation.json", {
                "protocol": self.protocol, "victim_seed": seed_stream(self.args.seed, "victim"),
                "clean_predictions_hash": array_hash(self.evaluations["clean"]),
                "metrics": metrics})


def main():
    global _OPTIONS
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--followup-protocol", dest="protocol", choices=[PROTOCOL, SELECTOR_PROTOCOL, FACTORIAL_PROTOCOL], default=PROTOCOL)
    parser.add_argument("--followup-selector", dest="selector", choices=["ae_pr", "pr", "ppr", "degree", "random"], required=True)
    parser.add_argument("--followup-global-transition", dest="global_transition", choices=["attribute", "topology"], required=True)
    parser.add_argument("--followup-local-graph", dest="local_graph", choices=["post", "pre"], default="post")
    parser.add_argument("--followup-artifacts", dest="artifacts", type=Path, required=True)
    _OPTIONS, remaining = parser.parse_known_args()
    validate_protocol_options(_OPTIONS)
    if "--oracle" in remaining:
        raise SystemExit("Oracle labels are prohibited in the follow-up protocol")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required (local Ollama: ollama)")
    remaining += ["--openai_api_key", api_key]
    # Unsupported/non-deterministic kernels raise instead of silently invalidating matching.
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    reset_rng(0)
    sys.argv = [str(ROOT / "meta.py"), *remaining]
    heir_module.Heirattack = FollowupAttack
    try:
        runpy.run_path(str(ROOT / "meta.py"), run_name="__main__")
    finally:
        heir_module.Heirattack = _BASE


if __name__ == "__main__":
    main()
