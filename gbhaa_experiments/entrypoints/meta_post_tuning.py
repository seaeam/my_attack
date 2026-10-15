"""Opt-in short-prefix tuning. Select configurations on validation nodes only."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN

import heir as heir_module
from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.action_state_protocol import ARMS, actions_hash, independent_restarts
from gbhaa_experiments.action_state_validation import validate_result as validate_parent
from gbhaa_experiments.entrypoints import meta_followup as followup_module
from gbhaa_experiments.entrypoints.meta_action_state import ActionStateAttack, BaseAttack
from gbhaa_experiments.followup_protocol import array_hash, as_array, graph_hash, reset_rng, seed_stream
from gbhaa_experiments.post_tuning_protocol import (
    PROTOCOL, get_profile, runtime_settings, operator_settings, score_nodes, GUIDED_SELECTOR_RULES, needs_surrogate,
)
from gbhaa_experiments.runner import write_json
from gbhaa_experiments import selector_mechanisms as mechanism


def validate_splits(n, train, validation, test):
    """Preserve disjoint labeled subsets; graph nodes may be outside all splits."""
    splits = {}
    for name, values in (("train", train), ("validation", validation), ("test", test)):
        values = np.asarray(values)
        if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
                or len(np.unique(values)) != len(values) or np.any(values < 0) or np.any(values >= n)):
            raise ValueError(f"Invalid {name} split")
        splits[name] = values.astype(np.int64, copy=True)
    combined = np.concatenate(list(splits.values()))
    # DeepRobust nettack's rounding can leave nodes outside every labeled split.
    # Keep the saved split IDs exactly: filling the gap would alter the protocol.
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Train, validation and test must be disjoint")
    return splits


class TuningAttack(ActionStateAttack):
    @property
    def protocol(self):
        return PROTOCOL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_profile(self.options.profile_id)
        # The inherited constructor verifies the legacy generator default first.
        # The cache wrapper reads this live value when a request is actually sent.
        if self.text_generator is None:
            raise ValueError("Post tuning requires text generation")
        self.text_generator.max_tokens = self.config["generation_max_tokens"]

    def configure_evaluation_split(self, train, validation, test):
        self.splits = validate_splits(self.nnodes, train, validation, test)

    def _check_options(self, target):
        for key, value in runtime_settings(self.config).items():
            if getattr(self.args, key, None) != value:
                raise ValueError(f"Tuning profile requires {key}={value!r}")
        if (not self.freeze_structure_features or self.use_oracle
                or self.text_attack_total_nodes is not None or target != self.options.max_actions
                or target < 1 or self.nnodes < 4 or 4*target > 10*self.nnodes):
            raise ValueError("Invalid frozen, non-oracle, four-node prefix budget")
        if self.text_generator.max_tokens != self.config["generation_max_tokens"]:
            raise ValueError("Generation token setting differs from profile")
        if not hasattr(self, "splits"):
            raise ValueError("Evaluation splits must be configured before replay")

    def input_metadata(self, features, adj, labels, idx_train, idx_eval, target):
        old = super().input_metadata(features, adj, labels, idx_train, idx_eval, target)
        source_job = dict(dataset=self.args.dataset, seed=self.args.seed,
                          budget=float(self.args.ptb_rate), variant="structure")
        parent = validate_parent(self.options.source_run/"result.json", source_job)
        trajectory = parent["trajectory"]
        for key in ("original_adj_hash", "original_features_hash", "training_nodes_hash",
                    "target_nodes_hash", "evaluation_labels_hash"):
            if old[key] != trajectory[key]:
                raise ValueError(f"Parent structural trajectory differs in {key}")
        if target > len(trajectory["actions"]):
            raise ValueError("Requested prefix exceeds the saved structural trajectory")
        if not np.array_equal(idx_train, self.splits["train"]) or not np.array_equal(
                idx_eval, np.union1d(self.splits["validation"], self.splits["test"])):
            raise ValueError("Replay splits differ from the configured evaluation partition")
        self._parent_record = parent
        edges = int(adj.nnz//2)
        old.update(protocol=PROTOCOL, budget=target/edges, effective_budget=target/edges,
            source_budget=float(self.args.ptb_rate), profile_id=self.options.profile_id,
            trial_id=self.options.trial_id, config=self.config,
            settings=runtime_settings(self.config), operator_settings=operator_settings(self.config),
            parent_fingerprint=parent["fingerprint"], parent_actions_hash=trajectory["actions_hash"],
            parent_protocol=parent["protocol"], prefix_length=int(target),
            validation_nodes_hash=array_hash(self.splits["validation"]),
            holdout_nodes_hash=array_hash(self.splits["test"]),
            generation_max_tokens=self.text_generator.max_tokens,
            structural_timing_scope="sum_of_parent_round_times")
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.trace_dir/"splits.npz", **self.splits)
        if needs_surrogate(self.config):
            old["selector_surrogate"] = self._prepare_selector_surrogate(features)
        if "editor_gain" in self.config:
            from gbhaa_experiments.projected_editing_runtime import prepare_editor
            old["editor_snapshot"] = prepare_editor(self,features=features,graph=adj,
                train_labels=as_array(labels)[self.splits['train']])
        return old

    def attack_features_with_text(self, *args, **kwargs):
        if "editor_gain" in self.config:
            from gbhaa_experiments.projected_editing_runtime import attack_features
            return attack_features(self, *args, **kwargs)
        return super().attack_features_with_text(*args, **kwargs)

    def _prepare_selector_surrogate(self, features):
        """Freeze clean predictions before replay; no non-training true labels."""
        if not needs_surrogate(self.config):
            return None
        output = self.surrogate.output.detach().clone()
        if (output.ndim != 2 or output.shape[0] != self.nnodes or output.shape[1] < 2
                or not torch.isfinite(output).all()):
            raise ValueError("Invalid clean selector surrogate output")
        self._selector_pseudo = output.argmax(dim=1).clone()
        self._selector_targets = np.setdiff1d(np.arange(self.nnodes, dtype=np.int64), self.splits["train"])
        self._selector_vocab_width = min(len(self.text_generator.vocab), features.shape[1])
        if not len(self._selector_targets) or not self._selector_vocab_width:
            raise ValueError("Guided selector needs non-training targets and a usable vocabulary")
        self._selector_surrogate_info = dict(
            model_class=type(self.surrogate).__module__+"."+type(self.surrogate).__qualname__,
            state_hashes={name: array_hash(value) for name, value in self.surrogate.state_dict().items()},
            clean_output_hash=array_hash(output), pseudo_labels_hash=array_hash(self._selector_pseudo),
            target_nodes_hash=array_hash(self._selector_targets), nclass=output.shape[1],
            vocabulary_width=self._selector_vocab_width,
            loss="mean NLL of fixed clean argmax pseudo-labels over all non-training nodes",
            edit_proxy="beneficial removals plus bounded unit additions in absent vocabulary columns")
        np.savez_compressed(self.trace_dir/"selector_pseudolabels.npz",
                            pseudo_labels=as_array(self._selector_pseudo), target_nodes=self._selector_targets,
                            clean_log_probabilities=as_array(output))
        return self._selector_surrogate_info

    def _rank_graph(self, features, graph, restarts):
        """Rank from this graph only; Pre never receives Post-derived candidates."""
        if "selector_rule" not in self.config:
            order, scores = score_nodes(features, graph, restarts, self.config, self._attacked_nodes)
            return order, scores, None
        states = mechanism.compute_states(features, graph, restarts, self.config)
        gains = None; loss = None
        if self.config["selector_rule"] in GUIDED_SELECTOR_RULES:
            adj = utils.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
            normalized = utils.normalize_adj_tensor(adj, sparse=True)
            gradient, loss = mechanism.feature_loss_gradient(
                self.surrogate, features, normalized, self._selector_pseudo, self._selector_targets)
            gains = mechanism.edit_potential(gradient, as_array(features), self._selector_vocab_width,
                                            self.config["text_max_added_words"])
        order, scores, info = mechanism.order_from_states(states, restarts, self._attacked_nodes,
            self.config["selector_rule"], self.config["selector_pool_size"], gains)
        trace = dict(version=mechanism.selector_version(self.config["selector_rule"]), rule=self.config["selector_rule"],
                     graph_hash=graph_hash(graph), restart_nodes=list(restarts), states=states.tolist(),
                     gains=None if gains is None else gains.tolist(), gradient_loss=loss, selection=info)
        return order, scores, trace

    def _load_plan(self):
        parent = self._parent_record["trajectory"]
        actions = parent["actions"][:self.options.max_actions]
        rounds = json.loads((self.options.source_run/"artifacts/rounds.json").read_text())["rounds"]
        duration = sum(float(r.get("structural_seconds", 0.)) for r in rounds[:len(actions)])
        return {"actions": actions, "actions_hash": actions_hash(actions), "structural_seconds": duration}

    def _select_round(self, features, pre, post, action):
        endpoints = action[:2]
        visits = self._attacked_nodes
        pre_order, pre_scores, pre_trace = self._rank_graph(features, pre, endpoints)
        self.full_adj = utils.sparse_mx_to_torch_sparse_tensor(post).to(self.device)
        post_order, post_scores, post_trace = self._rank_graph(features, post, endpoints)
        pre_nodes = self.filter_text_attack_nodes(pre_order)[:4]
        post_nodes = self.filter_text_attack_nodes(post_order)[:4]
        arm = self.options.phase
        restarts = endpoints
        if arm == "parallel":
            restarts = independent_restarts(self.nnodes, self.args.seed, self.round_index)
            order, _, parallel_trace = self._rank_graph(features, pre, restarts)
        else:
            order = post_order if arm == "endpoint_post" else pre_order
        eligible = self.filter_text_attack_nodes(order)
        nodes = eligible[:4]
        if len(nodes) != 4:
            raise ValueError("Cannot fill the four-node tuning workload")
        graph = post if arm == "endpoint_post" else pre
        if "editor_gain" in self.config:
            self._editor_graph = graph.copy()
        vectors = as_array(features)[endpoints]
        norm_product = np.prod(np.linalg.norm(vectors, axis=1))
        similarity = float(np.dot(*vectors)/norm_product) if norm_product > 1e-12 else 0.
        self.diagnostics[-1].update(pre_graph_hash=graph_hash(pre), post_graph_hash=graph_hash(post),
            diffusion_graph_hash=graph_hash(graph), local_graph="post" if arm == "endpoint_post" else "pre",
            local_edge_value=int(graph[action[0],action[1]]), restart_nodes=list(restarts),
            selected_nodes=nodes.tolist(), candidate_universe=self.nnodes, eligible_count=len(eligible),
            current_features_hash=array_hash(features), probe_pre_nodes=pre_nodes.tolist(),
            probe_post_nodes=post_nodes.tolist(),
            probe_jaccard=len(set(pre_nodes)&set(post_nodes))/len(set(pre_nodes)|set(post_nodes)),
            probe_score_l1=float(np.abs(pre_scores-post_scores).sum()),
            action_similarity=float(np.clip(similarity, 0, 1)), grouping_graph="post",
            grouping_graph_hash=graph_hash(post), selected_endpoint_count=len(set(nodes)&set(endpoints)),
            repeated_node_updates=sum(visits.get(int(n), 0) > 0 for n in nodes))
        if arm == "parallel":
            self.diagnostics[-1]["independent_restart_overlap"] = len(set(restarts)&set(endpoints))
        if pre_trace is not None:
            self.diagnostics[-1]["mechanism_traces"] = dict(pre=pre_trace, post=post_trace)
            if arm == "parallel":
                self.diagnostics[-1]["mechanism_traces"]["parallel"] = parallel_trace
        return nodes


def evaluate_validation(attack, original_features, original_adj, modified_features,
                        modified_adj, labels, victim_class=GCN):
    """Train using training labels only; never evaluate or save held-out ACC."""
    labels = torch.as_tensor(as_array(labels), dtype=torch.long)
    train = attack.splits["train"]; validation = attack.splits["validation"]
    training_labels = torch.zeros_like(labels)
    training_labels[train] = labels[train]
    nclass = getattr(attack, "nclass", int(labels[train].max())+1)
    metrics = {}; hashes = {}
    for stage, features, adj in (("clean_validation", original_features, original_adj),
                                  ("validation", modified_features, modified_adj)):
        reset_rng(seed_stream(attack.args.seed, "victim"))
        features = as_array(features)
        model = victim_class(nfeat=features.shape[1], nhid=attack.args.hidden, nclass=nclass,
                             dropout=attack.args.dropout, device=attack.device).to(attack.device)
        model.fit(features, adj, training_labels.numpy(), train)
        logits = as_array(model.output)[validation]
        y = as_array(labels)[validation]; predictions = logits.argmax(axis=1)
        metrics[stage+"_accuracy"] = float(np.mean(predictions == y))
        metrics[stage+"_nll"] = float(-logits[np.arange(len(y)), y].astype(np.float64).mean())
        hashes[stage] = array_hash(predictions)
        np.savez_compressed(attack.trace_dir/f"{stage}_predictions.npz", node_ids=validation,
                            labels=y, predictions=predictions, log_probabilities=logits)
    metrics["accuracy_drop_pp"] = 100*(metrics["clean_validation_accuracy"]-metrics["validation_accuracy"])
    result = dict(protocol=PROTOCOL, split="validation", victim_seed=seed_stream(attack.args.seed,"victim"),
        validation_nodes_hash=array_hash(validation), holdout_nodes_hash=array_hash(attack.splits["test"]),
        train_nodes_hash=array_hash(train), clean_predictions_hash=hashes["clean_validation"],
        validation_predictions_hash=hashes["validation"], target_count=len(validation), metrics=metrics)
    write_json(attack.trace_dir/"evaluation.json", result)
    print(f"Validation ACC={100*metrics['validation_accuracy']:.4f}%; NLL={metrics['validation_nll']:.6f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tune-profile", dest="profile_id", required=True)
    parser.add_argument("--tune-arm", dest="phase", choices=ARMS, required=True)
    parser.add_argument("--tune-artifacts", dest="artifacts", type=Path, required=True)
    parser.add_argument("--tune-source-run", dest="source_run", type=Path, required=True)
    parser.add_argument("--tune-cache", dest="cache", type=Path, required=True)
    parser.add_argument("--tune-max-actions", dest="max_actions", type=int, required=True)
    options, remaining = parser.parse_known_args()
    get_profile(options.profile_id)
    options.trial_id = f"{options.profile_id}__{options.phase}"
    options.protocol = PROTOCOL; options.selector = "ae_pr"; options.global_transition = "attribute"
    options.local_graph = "post" if options.phase == "endpoint_post" else "pre"
    followup_module._OPTIONS = options
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required (local Ollama: ollama)")
    remaining += ["--openai_api_key", key]
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1); reset_rng(0)
    sys.argv = [str(REPO_ROOT/"meta.py"), *remaining]
    heir_module.Heirattack = TuningAttack
    captured = io.StringIO()
    try:
        # Non-main run_name skips the legacy mixed validation/test evaluator.
        with contextlib.redirect_stdout(captured):
            namespace = runpy.run_path(str(REPO_ROOT/"meta.py"), run_name="__post_tuning_bootstrap__")
    finally:
        heir_module.Heirattack = BaseAttack
        for line in captured.getvalue().splitlines():
            if not line.startswith("Test set results:"):
                print(line)
    attack = namespace["model"]
    attack.configure_evaluation_split(namespace["idx_train"], namespace["idx_val"], namespace["idx_test"])
    attack.meta_attack_multi_step(namespace["features"], namespace["org_adj"], namespace["labels"],
        namespace["idx_train"], namespace["idx_unlabeled"], options.max_actions,
        n_step=1, type=namespace["args"].model)
    evaluate_validation(attack, namespace["features"], namespace["org_adj"],
                        attack.modified_features, attack.modified_adj, namespace["labels"])


if __name__ == "__main__":
    main()
