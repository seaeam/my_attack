import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import nn
from tqdm import tqdm
from deeprobust.graph import utils
from gb_division import gb_division
from deeprobust.graph.global_attack import BaseAttack
from gb_division_simple import GBCluster, KMeansCluster

# 文本攻击生成器导入
TEXT_ATTACK_IMPORT_ERROR = None
try:
    from text_attack_generator import TextAttackGenerator

    TEXT_ATTACK_AVAILABLE = True
except ImportError as exc:
    TEXT_ATTACK_AVAILABLE = False
    TEXT_ATTACK_IMPORT_ERROR = exc
    print("Warning: TextAttackGenerator not available. Text attack will be disabled.")


class Heirattack(BaseAttack):
    def __init__(
        self,
        model,
        nnodes,
        args,
        gb_data,
        features,
        feature_shape=None,
        attack_structure=True,
        attack_features=True,
        device="cpu",
        with_bias=False,
        lambda_=0.5,
        train_iters=10,
        lr=0.1,
        momentum=0.9,
        levels=2,
        use_oracle=False,
    ):
        requested_text_attack = getattr(args, "use_text_attack", False)
        if requested_text_attack and not TEXT_ATTACK_AVAILABLE:
            raise RuntimeError(
                "Text attack was requested, but TextAttackGenerator could not be "
                "imported. Install the text-attack dependencies or disable "
                "--use_text_attack."
            ) from TEXT_ATTACK_IMPORT_ERROR

        super(Heirattack, self).__init__(
            model, nnodes, attack_structure, attack_features, device
        )

        self.args = args
        self.gb_data = gb_data

        self.lambda_ = lambda_
        self.features = features

        assert (
            attack_features or attack_structure
        ), "attack_features or attack_structure cannot be both False"

        self.with_relu = model.with_relu

        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = Parameter(
            torch.FloatTensor(previous_size, self.nclass).to(device)
        )
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

        self.levels = levels
        self.M = int(self.nnodes ** (1.0 / self.levels))  # int(math.sqrt(N))
        self.coarsen_method = getattr(args, "coarsen_method", "gb").lower()
        self.gb_cluster = self._make_clusterer(self.M)
        self.use_oracle = use_oracle
        self.global_important_ratio = float(
            getattr(args, "global_important_ratio", 0.10)
        )
        self.global_ppr_alpha = float(getattr(args, "global_ppr_alpha", 0.15))
        self.global_ppr_iters = int(getattr(args, "global_ppr_iters", 30))
        self.global_seed_strategy = getattr(args, "global_seed_strategy", "uniform")
        self.freeze_structure_features = bool(
            getattr(args, "freeze_structure_features", False)
        )
        self.text_budget_per_node = int(getattr(args, "text_budget_per_node", 15))
        text_attack_nodes = getattr(args, "text_attack_nodes", None)
        self.text_attack_nodes = (
            None if text_attack_nodes is None else max(1, int(text_attack_nodes))
        )
        self.text_topk_ratio = float(getattr(args, "text_topk_ratio", 0.05))
        self.text_ppr_alpha = float(getattr(args, "text_ppr_alpha", 0.20))
        self.text_ppr_iters = int(getattr(args, "text_ppr_iters", 25))
        self.local_candidate_strategy = getattr(
            args, "local_candidate_strategy", "local_ae_ppr"
        ).lower()
        self.local_candidate_hops = int(getattr(args, "local_candidate_hops", 2))
        self.text_min_cluster_size = int(getattr(args, "text_min_cluster_size", 2))
        self.text_max_cluster_size = int(getattr(args, "text_max_cluster_size", 8))
        self.text_attack_max_visits = int(getattr(args, "text_attack_max_visits", 1))
        self.text_similarity_min = float(getattr(args, "text_similarity_min", 0.85))
        self.text_cdl_topk = int(getattr(args, "text_cdl_topk", 10))
        self.text_cluster_attr_topk = int(getattr(args, "text_cluster_attr_topk", 10))
        self.text_max_added_words = int(getattr(args, "text_max_added_words", 20))

        # 缓存
        self.template_cache = {}  # 簇模板缓存
        self.node2cluster = {}  # 节点到簇ID的映射
        self.local_candidate_rng = np.random.default_rng(getattr(args, "seed", None))

        # ========== 文本攻击生成器初始化 ==========
        self.use_text_attack = requested_text_attack
        if self.use_text_attack:
            llm_type = getattr(args, "llm_type", "gpt")

            # 根据LLM类型设置参数
            if llm_type == "llama":
                model_path = getattr(args, "llama_model_path", None)
                if model_path is None:
                    raise ValueError(
                        "llama_model_path is required when using llm_type=llama"
                    )

                self.text_generator = TextAttackGenerator(
                    dataset_name=getattr(args, "dataset", "cora"),
                    llm_type="llama",
                    model_path=model_path,
                    device=self.device,
                    feature_dim=self.nfeat,
                    allow_fallback_vocabulary=getattr(
                        args, "allow_fallback_vocabulary", False
                    ),
                    allow_partial_vocabulary=getattr(
                        args, "allow_partial_vocabulary", False
                    ),
                )
                print(
                    f"✅ Text attack generator initialized with local Llama model: {model_path}"
                )
            else:
                # GPT or other API-based models
                api_key = getattr(args, "openai_api_key", None)
                if api_key is None:
                    raise ValueError(
                        f"openai_api_key is required when using llm_type={llm_type}"
                    )

                self.text_generator = TextAttackGenerator(
                    dataset_name=getattr(args, "dataset", "cora"),
                    api_key=api_key,
                    base_url=getattr(args, "api_base_url", None),
                    device=self.device,
                    llm_type=llm_type,
                    feature_dim=self.nfeat,
                    allow_fallback_vocabulary=getattr(
                        args, "allow_fallback_vocabulary", False
                    ),
                    allow_partial_vocabulary=getattr(
                        args, "allow_partial_vocabulary", False
                    ),
                    num_retries=getattr(args, "text_retries", 1),  # 默认只重试1次
                )
                print(
                    f"✅ Text attack generator initialized with {llm_type.upper()} API"
                )
        else:
            self.text_generator = None

    def _make_clusterer(self, n_clusters, args=None):
        args = self.args if args is None else args
        method = getattr(args, "coarsen_method", "gb").lower()
        seed = getattr(args, "seed", None)

        if method == "gb":
            return GBCluster(n_clusters=n_clusters, mode="euclidean", verbose=0)
        if method == "kmeans":
            return KMeansCluster(
                n_clusters=n_clusters,
                mode="euclidean",
                verbose=0,
                random_state=seed,
            )

        raise ValueError(f"Unsupported coarsen_method: {method}")

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1.0 / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1.0 / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def _compute_node_contexts(self, full_features, adj, U):
        """为节点计算上下文特征（用于门控网络）"""
        device = self.device
        U_t = torch.from_numpy(np.asarray(U)).long().to(device)

        # 1. 基础特征
        X_u = full_features[U_t]  # [m, d]

        # 2. 度数特征（标准化）
        degrees = np.asarray(adj.sum(axis=1)).reshape(-1)[U]  # [m]
        deg_feat = torch.from_numpy(degrees).float().to(device).unsqueeze(1)  # [m, 1]
        deg_feat = (deg_feat - deg_feat.mean()) / (deg_feat.std() + 1e-8)

        # 3. GB 簇特征（如果可用）
        if hasattr(self, "node2gb") and self.node2gb is not None:
            gb_ids = self.node2gb[U]  # [m]
            gb_onehot = torch.zeros(len(U), max(gb_ids.max() + 1, 1)).to(device)
            gb_onehot[torch.arange(len(U)), torch.from_numpy(gb_ids).long()] = 1.0
            # 降维到合适尺寸
            if gb_onehot.size(1) > 10:
                gb_feat = gb_onehot[:, :10]
            else:
                gb_feat = F.pad(gb_onehot, (0, max(0, 10 - gb_onehot.size(1))))
        else:
            gb_feat = torch.zeros(len(U), 10).to(device)

        # 拼接上下文
        # 为了匹配 self.nfeat，我们投影额外特征
        extra_feat = torch.cat([deg_feat, gb_feat], dim=1)  # [m, 11]
        if extra_feat.size(1) < X_u.size(1):
            padding = torch.zeros(len(U), X_u.size(1) - extra_feat.size(1)).to(device)
            extra_feat = torch.cat([extra_feat, padding], dim=1)
        else:
            extra_feat = extra_feat[:, : X_u.size(1)]

        # 加权融合：主要保留节点特征
        context = 0.9 * X_u + 0.1 * extra_feat
        return context

    # 为未标记节点生成伪标签，用于自训练
    def self_training_label(self, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        if self.use_oracle:
            return labels
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training

    # 在每次攻击决策前，使用当前扰动下的特征和邻接矩阵，训练一次轻量级代理分类器
    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(
                loss_labeled, self.weights, create_graph=True
            )
            self.w_velocities = [
                self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)
            ]
            if self.with_bias:
                bias_grads = torch.autograd.grad(
                    loss_labeled, self.biases, create_graph=True
                )
                self.b_velocities = [
                    self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)
                ]

            self.weights = [
                w - self.lr * v for w, v in zip(self.weights, self.w_velocities)
            ]
            if self.with_bias:
                self.biases = [
                    b - self.lr * v for b, v in zip(self.biases, self.b_velocities)
                ]

    # 用当前代理网络的参数，对输入特征做前向传播，输出每个节点的隐藏层嵌入,用于后续的聚类分组和攻击决策
    def get_embeddings(self, features, adj_norm):
        hidden = features
        for ix, w in enumerate(self.weights):
            if ix >= len(self.weights) - 1:
                break
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b

            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)
        return hidden

    # 在当前代理网络参数下，计算攻击损失对结构（邻接矩阵）和特征的梯度，用于指导后续的扰动选择。
    def get_meta_grad(self, features, adj_norm, labels, labels_u):
        """
        返回 attack_loss 分别对 self.cur_adj（结构）和输入 features（特征）的梯度。
        不再依赖 self.feature_changes。
        """
        # 如果需要对特征求梯度，克隆一份并打开 requires_grad
        if self.attack_features:
            X = features.detach().clone()
            X.requires_grad_(True)
        else:
            X = features

        hidden = X
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        attack_loss = torch.sum(-output * labels) / (torch.sum(labels) + 1e-8)
        attack_loss += torch.sum(-output * labels_u) / (torch.sum(labels_u) + 1e-8)

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(
                attack_loss, self.cur_adj, retain_graph=True
            )[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, X, retain_graph=True)[0]

        return adj_grad, feature_grad

    def _merge_small_text_clusters(
        self, target_nodes, cluster_ids, target_embeddings, min_cluster_size=2
    ):
        target_nodes = np.asarray(target_nodes, dtype=np.int64)
        cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
        target_embeddings = np.asarray(target_embeddings, dtype=np.float32)

        if target_nodes.size == 0:
            return {}

        raw_clusters = {}
        for pos, cid in enumerate(cluster_ids.tolist()):
            raw_clusters.setdefault(int(cid), []).append(pos)

        if len(raw_clusters) <= 1:
            return {0: target_nodes.tolist()}

        large_clusters = {
            cid: positions
            for cid, positions in raw_clusters.items()
            if len(positions) >= min_cluster_size
        }
        small_clusters = {
            cid: positions
            for cid, positions in raw_clusters.items()
            if len(positions) < min_cluster_size
        }

        if not large_clusters:
            return {0: target_nodes.tolist()}

        center_ids = sorted(large_clusters.keys())
        centers = np.stack(
            [target_embeddings[large_clusters[cid]].mean(axis=0) for cid in center_ids],
            axis=0,
        )

        for positions in small_clusters.values():
            points = target_embeddings[positions]
            if self.gb_cluster.mode == "cosine":
                points_norm = points / (
                    np.linalg.norm(points, axis=1, keepdims=True) + 1e-12
                )
                centers_norm = centers / (
                    np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
                )
                dists = 1.0 - (points_norm @ centers_norm.T)
            else:
                dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)

            nearest_center = dists.argmin(axis=1)
            for pos, center_idx in zip(positions, nearest_center.tolist()):
                large_clusters[center_ids[center_idx]].append(pos)

        merged_clusters = {}
        for new_cid, cid in enumerate(sorted(large_clusters.keys())):
            positions = sorted(large_clusters[cid])
            merged_clusters[new_cid] = target_nodes[positions].tolist()

        return merged_clusters

    def _fallback_partition_text_cluster(
        self, target_nodes, target_embeddings, min_cluster_size=2, max_cluster_size=8
    ):
        target_nodes = np.asarray(target_nodes, dtype=np.int64)
        target_embeddings = np.asarray(target_embeddings, dtype=np.float32)
        n = target_nodes.size

        if n <= 1 or n <= max_cluster_size:
            return [target_nodes.tolist()]

        max_parts = max(1, n // max(1, min_cluster_size))
        num_parts = max(2, int(math.ceil(n / float(max_cluster_size))))
        num_parts = min(num_parts, max_parts)
        if num_parts <= 1:
            return [target_nodes.tolist()]

        centered = target_embeddings - target_embeddings.mean(axis=0, keepdims=True)
        try:
            if np.linalg.norm(centered) > 1e-8:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                scores = centered @ vh[0]
            else:
                scores = np.arange(n, dtype=np.float32)
        except np.linalg.LinAlgError:
            scores = np.arange(n, dtype=np.float32)

        order = np.argsort(scores)

        while num_parts > 1:
            base = n // num_parts
            remainder = n % num_parts
            part_sizes = [
                base + (1 if part_idx < remainder else 0)
                for part_idx in range(num_parts)
            ]
            if min(part_sizes) >= min_cluster_size:
                break
            num_parts -= 1

        if num_parts <= 1:
            return [target_nodes.tolist()]

        base = n // num_parts
        remainder = n % num_parts
        part_sizes = [
            base + (1 if part_idx < remainder else 0) for part_idx in range(num_parts)
        ]

        partitions = []
        start = 0
        for size in part_sizes:
            idx = order[start : start + size]
            partitions.append(target_nodes[idx].tolist())
            start += size

        return [part for part in partitions if part]

    def _split_text_cluster_once(
        self, target_nodes, target_embeddings, min_cluster_size=2, max_cluster_size=8
    ):
        target_nodes = np.asarray(target_nodes, dtype=np.int64)
        target_embeddings = np.asarray(target_embeddings, dtype=np.float32)
        n = target_nodes.size

        if n <= 1 or n <= max_cluster_size:
            return [target_nodes.tolist()]

        max_clusters = max(1, n // max(1, min_cluster_size))
        target_k = max(2, int(math.ceil(n / float(max_cluster_size))))
        target_k = min(target_k, max_clusters)

        if target_k <= 1:
            return [target_nodes.tolist()]

        local_clusterer = self._make_clusterer(n_clusters=target_k)
        cluster_ids = local_clusterer.fit_predict(target_embeddings)
        if isinstance(cluster_ids, torch.Tensor):
            cluster_ids = cluster_ids.detach().cpu().numpy()

        merged_clusters = self._merge_small_text_clusters(
            target_nodes,
            cluster_ids,
            target_embeddings,
            min_cluster_size=min_cluster_size,
        )
        cluster_list = [nodes for nodes in merged_clusters.values() if nodes]

        if len(cluster_list) <= 1:
            return self._fallback_partition_text_cluster(
                target_nodes,
                target_embeddings,
                min_cluster_size=min_cluster_size,
                max_cluster_size=max_cluster_size,
            )

        largest_child = max(len(nodes) for nodes in cluster_list)
        if largest_child >= n:
            return self._fallback_partition_text_cluster(
                target_nodes,
                target_embeddings,
                min_cluster_size=min_cluster_size,
                max_cluster_size=max_cluster_size,
            )

        return cluster_list

    def _build_round_text_clusters(
        self, target_nodes, target_embeddings, min_cluster_size=2, max_cluster_size=8
    ):
        target_nodes = np.asarray(target_nodes, dtype=np.int64)
        if target_nodes.size == 0:
            return {}

        if isinstance(target_embeddings, torch.Tensor):
            embeddings_np = target_embeddings.detach().cpu().float().numpy()
        else:
            embeddings_np = np.asarray(target_embeddings, dtype=np.float32)

        if embeddings_np.ndim != 2 or embeddings_np.shape[0] != target_nodes.size:
            raise ValueError("target_embeddings must align with target_nodes")

        n = target_nodes.size
        if n <= 1:
            return {0: target_nodes.tolist()}
        if n <= max_cluster_size:
            return {0: target_nodes.tolist()}
        final_clusters = {}
        next_cid = 0
        pending = [(target_nodes, embeddings_np)]

        while pending:
            cur_nodes, cur_embeddings = pending.pop(0)
            cur_nodes = np.asarray(cur_nodes, dtype=np.int64)
            cur_embeddings = np.asarray(cur_embeddings, dtype=np.float32)
            cur_size = cur_nodes.size

            if cur_size <= 1 or cur_size <= max_cluster_size:
                final_clusters[next_cid] = cur_nodes.tolist()
                next_cid += 1
                continue

            if cur_size < 2 * min_cluster_size:
                final_clusters[next_cid] = cur_nodes.tolist()
                next_cid += 1
                continue

            child_clusters = self._split_text_cluster_once(
                cur_nodes,
                cur_embeddings,
                min_cluster_size=min_cluster_size,
                max_cluster_size=max_cluster_size,
            )

            if len(child_clusters) <= 1 and len(child_clusters[0]) == cur_size:
                final_clusters[next_cid] = cur_nodes.tolist()
                next_cid += 1
                continue

            node_to_pos = {
                int(node): pos for pos, node in enumerate(cur_nodes.tolist())
            }
            for child_nodes in child_clusters:
                child_nodes = np.asarray(child_nodes, dtype=np.int64)
                if child_nodes.size == 0:
                    continue

                child_positions = np.asarray(
                    [node_to_pos[int(node)] for node in child_nodes], dtype=np.int64
                )
                child_embeddings = cur_embeddings[child_positions]

                if (
                    child_nodes.size > max_cluster_size
                    and child_nodes.size >= 2 * min_cluster_size
                ):
                    pending.append((child_nodes, child_embeddings))
                else:
                    final_clusters[next_cid] = child_nodes.tolist()
                    next_cid += 1

        return final_clusters

    def _select_discriminative_words(
        self, cluster_center, full_features, cluster_nodes, topk=10
    ):
        """
        选取跨类混淆词：与簇中心差异最大方向上的词汇。
        从全图特征均值和簇中心的差异中选取 top-k 个维度对应词汇。
        """
        if self.text_generator is None:
            return []

        # 全图特征均值
        if torch.is_tensor(full_features):
            feat_np = full_features.detach().cpu()
            if feat_np.is_sparse:
                feat_np = feat_np.to_dense()
            feat_np = feat_np.numpy()
        else:
            feat_np = np.asarray(full_features)

        global_mean = feat_np.mean(axis=0)

        # 优先注入簇内缺失但全图常见的词，减少“只复述本簇属性”的保守扰动。
        vocab = self.text_generator.vocab
        n_vocab = min(len(vocab), len(cluster_center), len(global_mean))
        if n_vocab <= 0:
            return []
        missing_scores = np.maximum(global_mean[:n_vocab] - cluster_center[:n_vocab], 0)
        if np.max(missing_scores) <= 1e-12:
            missing_scores = np.abs(cluster_center[:n_vocab] - global_mean[:n_vocab])
        top_indices = np.argsort(-missing_scores)[:topk]
        discriminative_words = [vocab[i] for i in top_indices if i < len(vocab)]
        return discriminative_words

    def attack_features_with_text(
        self,
        target_nodes,
        full_features,
        labels_st,
        budget_per_node=20,
        target_embeddings=None,
        text_clusters=None,
    ):
        """
        Cluster-based Text Attack Generation
        """
        if not self.use_text_attack:
            return full_features
        if self.text_generator is None:
            raise RuntimeError(
                "Text attack was requested, but the text generator is not initialized."
            )

        modified_features = full_features.clone()

        if text_clusters is not None:
            clusters = text_clusters
        elif target_embeddings is not None:
            clusters = self._build_round_text_clusters(
                target_nodes,
                target_embeddings,
                min_cluster_size=self.text_min_cluster_size,
                max_cluster_size=self.text_max_cluster_size,
            )
        else:
            clusters = {}
            for node in target_nodes:
                cid = self.node2cluster.get(node, -1)
                if cid not in clusters:
                    clusters[cid] = []
                clusters[cid].append(node)

        cluster_sizes = [len(nodes) for nodes in clusters.values()]
        if cluster_sizes:
            avg_size = sum(cluster_sizes) / float(len(cluster_sizes))
            print(
                "🔥 Text Attack: "
                f"{len(target_nodes)} nodes grouped into {len(clusters)} clusters "
                f"(min/avg/max = {min(cluster_sizes)}/{avg_size:.1f}/{max(cluster_sizes)})."
            )
        else:
            print(f"🔥 Text Attack: {len(target_nodes)} nodes grouped into 0 clusters.")

        # Stats
        targets = len(target_nodes)
        template_attempts = 0
        template_successes = 0
        template_failures = 0
        cache_hits = 0
        feature_writes = 0
        feature_changes = 0
        node_failures = 0
        cluster_failures = 0

        # 2. Process each cluster
        for cid, nodes in tqdm(clusters.items(), desc="Cluster Attack"):
            try:
                # Prepare cluster signature
                # Get cluster center feature
                nodes_tensor = torch.tensor(nodes).long().to(self.device)
                cluster_feat = full_features[nodes_tensor]
                if cluster_feat.is_sparse:
                    cluster_feat = cluster_feat.to_dense()

                cluster_center = cluster_feat.mean(dim=0).cpu().numpy()

                # Extract cluster attributes (top words)
                (
                    cluster_attrs,
                    _,
                ) = self.text_generator.extract_words_from_bow_vector(cluster_center)
                cluster_attrs = cluster_attrs[: self.text_cluster_attr_topk]

                # Discriminative words: 跨类混淆词（从非簇中心方向选取）
                discriminative_words = self._select_discriminative_words(
                    cluster_center, full_features, nodes, topk=self.text_cdl_topk
                )

                # Cache Key: Cluster ID + Attributes hash
                signature = f"{cid}_{hash(tuple(sorted(cluster_attrs)))}"

                # Check Cache
                if signature in self.template_cache:
                    templates = self.template_cache[signature]
                    if not templates:
                        raise RuntimeError(
                            f"Cached template batch is empty for cluster {cid}"
                        )
                    cache_hits += 1
                else:
                    # Select Representative Node (Highest PPR score)
                    if hasattr(self, "global_ppr_scores"):
                        rep_node = max(nodes, key=lambda x: self.global_ppr_scores[x])
                    else:
                        rep_node = nodes[0]  # Fallback

                    # Generate Templates
                    template_attempts += 1
                    try:
                        templates = self.text_generator.generate_cluster_template(
                            cluster_attributes=cluster_attrs,
                            discriminative_words=discriminative_words,
                            num_candidates=3,
                        )
                        if not templates:
                            raise RuntimeError(
                                f"Generated template batch is empty for cluster {cid}"
                            )
                    except Exception:
                        template_failures += 1
                        raise
                    template_successes += 1

                    # Cache
                    self.template_cache[signature] = templates

                # 3. Adapt for each node
                for node in nodes:
                    try:
                        # Pick a template
                        template = np.random.choice(templates)

                        # Node specific requirements
                        current_bow = full_features[node].detach().cpu()
                        if current_bow.is_sparse:
                            current_bow = current_bow.to_dense().numpy()
                        else:
                            current_bow = current_bow.numpy()

                        used_words, _ = (
                            self.text_generator.extract_words_from_bow_vector(
                                current_bow
                            )
                        )
                        used_words = used_words[:budget_per_node]
                        force_words = list(
                            dict.fromkeys(
                                discriminative_words[: self.text_cdl_topk]
                                + used_words
                            )
                        )

                        # Lightweight Adaptation: explicitly force confusing words
                        # plus node-specific words into the vectorized text.
                        missing = [
                            w for w in force_words if w.lower() not in template.lower()
                        ]

                        adapted_text = template
                        if missing:
                            # Append simple phrase to fill missing words
                            adapted_text += (
                                f" Key aspects include: {', '.join(missing)}."
                            )

                        # Vectorize and Write back
                        new_bow = self.text_generator.vectorizer.transform(
                            [adapted_text]
                        ).toarray()[0]

                        # Align dimensions
                        feat_dim = modified_features.shape[1]
                        if new_bow.shape[0] != feat_dim:
                            aligned = np.zeros(feat_dim, dtype=np.float32)
                            copy_len = min(new_bow.shape[0], feat_dim)
                            aligned[:copy_len] = new_bow[:copy_len]
                            new_bow = aligned

                        # --- text_max_added_words: 限制新增词条数 ---
                        orig_nonzero = set(np.where(current_bow > 0)[0])
                        new_nonzero = np.where(new_bow > 0)[0]
                        added_indices = [
                            idx for idx in new_nonzero if idx not in orig_nonzero
                        ]
                        if len(added_indices) > self.text_max_added_words:
                            # 保留 new_bow 值最大的 text_max_added_words 个新增词
                            added_vals = new_bow[added_indices]
                            keep_idx = np.argsort(-added_vals)[
                                : self.text_max_added_words
                            ]
                            keep_set = set(np.array(added_indices)[keep_idx])
                            for idx in added_indices:
                                if idx not in keep_set:
                                    new_bow[idx] = 0.0

                        # --- text_similarity_min: 相似度约束投影 ---
                        orig_vec = current_bow.astype(np.float32)
                        new_vec = new_bow.astype(np.float32)
                        orig_norm = np.linalg.norm(orig_vec)
                        new_norm = np.linalg.norm(new_vec)
                        if orig_norm > 1e-12 and new_norm > 1e-12:
                            cos_sim = np.dot(orig_vec, new_vec) / (orig_norm * new_norm)
                            if cos_sim < self.text_similarity_min:
                                # 线性插值回原始向量直到满足相似度下限
                                lo, hi = 0.0, 1.0
                                for _ in range(20):  # 二分查找
                                    mid = (lo + hi) / 2.0
                                    blended = mid * new_vec + (1.0 - mid) * orig_vec
                                    b_norm = np.linalg.norm(blended)
                                    if b_norm < 1e-12:
                                        break
                                    sim_b = np.dot(orig_vec, blended) / (
                                        orig_norm * b_norm
                                    )
                                    if sim_b >= self.text_similarity_min:
                                        lo = mid
                                    else:
                                        hi = mid
                                new_bow = lo * new_vec + (1.0 - lo) * orig_vec

                        feature_changed = not np.allclose(new_bow, current_bow)
                        modified_features[node] = (
                            torch.from_numpy(new_bow).float().to(self.device)
                        )
                        feature_writes += 1
                        if feature_changed:
                            feature_changes += 1

                    except Exception as e:
                        node_failures += 1
                        print(f"Error processing text attack node {node}: {e}")

            except Exception as e:
                cluster_failures += 1
                print(f"Error processing cluster {cid}: {e}")
                continue

        completed = int(
            template_successes + cache_hits > 0
            and feature_writes > 0
            and feature_changes > 0
        )
        print(
            "Text Attack Completion: "
            f"targets={targets}, template_attempts={template_attempts}, "
            f"template_successes={template_successes}, "
            f"template_failures={template_failures}, cache_hits={cache_hits}, "
            f"feature_writes={feature_writes}, feature_changes={feature_changes}, "
            f"node_failures={node_failures}, cluster_failures={cluster_failures}, "
            f"completed={completed}"
        )
        print(
            f"✅ Cluster Attack Done. Success: {feature_writes}/{targets}, "
            f"LLM Calls: {template_attempts}, Cache Hits: {cache_hits}"
        )
        if not completed:
            raise RuntimeError(
                "Text attack did not produce at least one actual feature change."
            )
        return modified_features

    # 层次化图对抗攻击的多步元攻击主函数
    # 实现基于粗化图的多步元攻击，通过递归聚类和梯度指导来选择最优扰动
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

        # labels_st = self.self_training_label(labels, idx_train)
        # labels_onehot = torch.zeros(self.nnodes, self.nclass).to(self.device)
        # if 'Both' in type or 'Self' in type:
        #     labels_onehot[idx_unlabeled] = labels_onehot[idx_unlabeled].scatter_(1, labels_st[idx_unlabeled].unsqueeze(1), 1) / len(idx_unlabeled)
        # if 'Both' in type or 'Train' in type:
        #     labels_onehot[idx_train] = labels_onehot[idx_train].scatter_(1, labels_st[idx_train].unsqueeze(1), 1) / len(idx_train)

        n_turns = math.ceil(n_perturbations * 1.0 / n_step)
        tot_perturbs = 0
        full_adj_cpu = ori_adj.copy()
        added, deled = sp.csr_matrix(ori_adj.shape), sp.csr_matrix(ori_adj.shape)

        num_add, num_del, depth = 0, 0, 0

        for i in tqdm(range(n_turns), desc="Perturbing graph"):
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

            # === 全局属性增强 PageRank：找出重要节点 ===
            # 计算全图的属性增强 PPR 分数
            self.global_ppr_scores = self.compute_global_ppr(
                fea=search_features,
                adj=full_adj_cpu,
                topk_ratio=self.global_important_ratio,
                alpha=self.global_ppr_alpha,
                N=self.global_ppr_iters,
                use_cos_sim=True,
                seed_strategy=self.global_seed_strategy,
            )
            # 全局重要节点索引（按 PPR 分数排序）
            global_k = min(
                self.nnodes,
                max(self.M, int(round(self.global_important_ratio * self.nnodes))),
            )
            global_important_nodes = np.argsort(-self.global_ppr_scores)[:global_k]

            # GB Division (粒球划分) - 使用重要节点引导
            pool = [range(self.nnodes)]
            childs = [[]]
            parents = [-1]
            levels = [1]
            cur = 0
            while cur < len(pool):
                subgraph = pool[cur]
                n = len(subgraph)

                if n <= self.M:  # or levels[cur] >= self.levels: # Terminals
                    if len(subgraph) > 1:
                        for i in subgraph:
                            childs[cur].append(len(pool))
                            pool.append([i])
                            childs.append([])
                            parents.append(cur)
                            levels.append(levels[cur] + 1)
                    cur += 1
                    continue

                # 使用粒球划分（基于嵌入空间的简化版本）
                subgraph_list = list(subgraph)

                # 找出子图中的重要节点（与全局重要节点的交集）
                subgraph_set = set(subgraph_list)
                targets = [
                    node for node in global_important_nodes if node in subgraph_set
                ]

                # 使用重要节点引导聚类
                m = len(targets)
                if m >= self.M:
                    # 如果重要节点数量足够，先对它们聚类得到初始中心
                    cid = self.gb_cluster.fit_predict(embeddings[targets, :])
                    # 然后用这些中心对整个子图进行分配
                    cid = self.gb_cluster.fit_predict(
                        embeddings[subgraph_list, :],
                        centroids=self.gb_cluster.centroids,
                    )
                else:
                    # 重要节点不足，直接对整个子图聚类
                    cid = self.gb_cluster.fit_predict(embeddings[subgraph_list, :])

                for i in range(self.M):
                    childs[cur].append(len(pool))
                    pool.append([])
                    childs.append([])
                    parents.append(cur)
                    levels.append(levels[cur] + 1)

                for i in range(n):
                    id = cid[i].item()
                    pool[-id - 1].append(subgraph[i])

                cur += 1

            # 构建 node2cluster 映射 (使用叶子节点作为簇)
            self.node2cluster = {}
            for i in range(len(pool)):
                # 如果是叶子节点（没有孩子）且包含节点
                if len(childs[i]) == 0 and len(pool[i]) > 0:
                    for node_idx in pool[i]:
                        self.node2cluster[node_idx] = i

            for step in range(n_step):
                tot_perturbs += 1
                if tot_perturbs > n_perturbations:
                    break

                while len((full_adj_cpu - ori_adj).nonzero()[0]) < tot_perturbs * 2:
                    # print((full_adj_cpu - ori_adj).nonzero()[0].shape)
                    # Root = 0
                    inpool_set = set(childs[0])
                    status = "unknown"
                    targetI, targetJ = 0, 0

                    # while status == 'unknown' or len(childs[targetI]) > 0 or len(childs[targetJ]) > 0:
                    while True:  # for level in range(self.levels):
                        depth += 1

                        inpool = list(inpool_set)
                        n = len(inpool)
                        adj_inpool = np.zeros((n, n))
                        added_inpool = np.zeros((n, n))
                        deled_inpool = np.zeros((n, n))
                        feature_inpool = torch.zeros((n, self.nfeat)).to(self.device)
                        # labels_inpool = torch.zeros((n, self.nclass)).to(self.device)
                        labels_inpool_l = torch.zeros((n, self.nclass)).to(self.device)
                        labels_inpool_ul = torch.zeros((n, self.nclass)).to(self.device)

                        sizes = torch.zeros((n, 1)).to(self.device)
                        cids = np.zeros(self.nnodes)

                        for i in range(n):
                            nodes = pool[inpool[i]]
                            sizes[i] = len(nodes)
                            # adj_inpool[i][i] = sizes[i] - 1
                            feature_inpool[i] = torch.mean(
                                search_features[nodes, :], dim=0
                            )
                            # labels_inpool[i] = torch.sum(labels_onehot[nodes, :], dim=0)
                            if "Both" in type or "Train" in type:
                                labels_inpool_l[i] = torch.sum(
                                    labels_oh_l[nodes, :], dim=0
                                )
                            if "Both" in type or "Self" in type:
                                labels_inpool_ul[i] = torch.sum(
                                    labels_oh_ul[nodes, :], dim=0
                                )
                            for j in nodes:
                                cids[j] = i
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
                        # best_score, best_status, I, J = -1e10, 'unknown', -1, -1

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
                                            score = adj_grad[i][
                                                j
                                            ]  # / (possible_edges - adj_inpool[i][j])
                                            if score > best_score:
                                                best_score, best_status, I, J = (
                                                    score,
                                                    "add",
                                                    i,
                                                    j,
                                                )
                                    if status in ["unknown", "del"]:
                                        if adj_inpool[i][j] > added_inpool[i][j]:
                                            score = -adj_grad[i][
                                                j
                                            ]  # / adj_inpool[i][j]
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
                            for i in childs[inpool[I]]:
                                if len(pool[i]) > 0:
                                    inpool_set.add(i)
                        if sizes[J] > 1 and not I == J:
                            inpool_set.remove(inpool[J])
                            for i in childs[inpool[J]]:
                                if len(pool[i]) > 0:
                                    inpool_set.add(i)

                    # 步骤4.5.9: 执行实际的扰动操作

                    # 1) 刚刚选出的两个具体节点
                    row_idx, col_idx = pool[targetI][0], pool[targetJ][0]

                    # ===== 文本/特征攻击（仅在 attack_features 开启时执行）=====
                    if self.attack_features:
                        # ===== 文本攻击：根据结构扰动端点选择属性攻击候选节点 =====
                        U = self.select_text_candidate_nodes(
                            fea=full_features,
                            adj=full_adj_cpu,
                            row_idx=row_idx,
                            col_idx=col_idx,
                        )

                        # 使用文本生成方法攻击 PPR 邻居节点 - 只攻击未攻击过的节点
                        if self.use_text_attack and self.text_generator is not None:
                            # 过滤掉已达到最大访问次数的节点
                            if not hasattr(self, "_attacked_nodes"):
                                self._attacked_nodes = {}  # node -> visit_count

                            new_nodes = np.array(
                                [
                                    n
                                    for n in U
                                    if self._attacked_nodes.get(n, 0)
                                    < self.text_attack_max_visits
                                ]
                            )

                            if len(new_nodes) > 0:
                                print(
                                    f"\n🎯 [Step {tot_perturbs}/{n_perturbations}] "
                                    f"Text attack: {len(new_nodes)} new nodes (skipped {len(U)-len(new_nodes)} already attacked)"
                                )

                                text_embeddings = self.get_embeddings(
                                    full_features, adj_norm
                                )
                                new_nodes_tensor = (
                                    torch.from_numpy(new_nodes)
                                    .long()
                                    .to(text_embeddings.device)
                                )
                                round_text_embeddings = text_embeddings[
                                    new_nodes_tensor
                                ]
                                text_clusters = self._build_round_text_clusters(
                                    new_nodes,
                                    round_text_embeddings,
                                    min_cluster_size=self.text_min_cluster_size,
                                    max_cluster_size=self.text_max_cluster_size,
                                )

                                # 执行文本攻击 - 只攻击新节点
                                full_features = self.attack_features_with_text(
                                    target_nodes=new_nodes,
                                    full_features=full_features,
                                    labels_st=labels_st,
                                    budget_per_node=self.text_budget_per_node,
                                    target_embeddings=round_text_embeddings,
                                    text_clusters=text_clusters,
                                )

                                # 记录已攻击的节点（增加计数）
                                for nd in new_nodes.tolist():
                                    self._attacked_nodes[nd] = (
                                        self._attacked_nodes.get(nd, 0) + 1
                                    )
                            else:
                                print(
                                    f"\n⏭️ [Step {tot_perturbs}/{n_perturbations}] "
                                    f"All {len(U)} nodes already attacked, skipping"
                                )

                    # 2) 执行扰动
                    full_adj_cpu[row_idx, col_idx] = 1 - full_adj_cpu[row_idx, col_idx]
                    full_adj_cpu[col_idx, row_idx] = 1 - full_adj_cpu[col_idx, row_idx]

                    # 记录扰动统计信息
                    if status == "add":
                        num_add += 1
                        added[row_idx, col_idx] = 1
                        added[col_idx, row_idx] = 1
                    else:
                        num_del += 1
                        deled[row_idx, col_idx] = 1
                        deled[col_idx, row_idx] = 1

        # 步骤5: 输出攻击结果和统计信息
        if self.attack_structure:
            self.modified_adj = full_adj_cpu
        if self.attack_features:
            # 文本攻击已在循环中完成，直接保存最终特征
            self.modified_features = full_features.detach()
            feature_status = (
                "text attacks executed in-loop"
                if self.use_text_attack and self.text_generator is not None
                else "text attack disabled; features unchanged"
            )
            print(
                f"\n✅ Attack completed: Structure perturbations={num_add+num_del}, {feature_status}"
            )
        else:
            self.modified_features = full_features.detach()
            print(
                f"\n✅ Attack completed: Structure perturbations={num_add+num_del} (edge-only, feature attack disabled)"
            )

    def ppr_topk_from_seed(
        self,
        fea: torch.Tensor,
        adj: sp.spmatrix,
        seed_u: int,
        topk_ratio: float = 0.05,
        alpha: float = 0.2,
        N: int = 25,
        eps: float = 1e-12,
        use_cos_sim: bool = True,
    ):
        """
        仅计算从 seed_u 出发的属性增强 PPR，并返回 Top-k 节点与完整分数。
        - fea: [n, d] torch.Tensor（在 CPU 或 GPU 均可，内部会转 CPU）
        - adj: scipy.sparse (n, n) 原始邻接（建议无向）
        - seed_u: 起点节点索引 (0..n-1)
        - use_cos_sim: 用特征余弦相似度给边加权；若为 False，则直接用结构邻接归一化
        返回:
        - topk_idx: np.ndarray[int]，Top-k 节点索引
        - scores: np.ndarray[float]，长度 n 的 PPR 分数向量
        """
        n = adj.shape[0]
        assert 0 <= seed_u < n
        A = adj.tocsr().astype(np.float32)
        A.setdiag(0)
        A.eliminate_zeros()
        # 仅在现有边上计算相似度
        if use_cos_sim:
            fea_cpu = fea.detach().to("cpu").float()
            # 处理稀疏张量：转为稠密
            if fea_cpu.is_sparse:
                fea_cpu = fea_cpu.to_dense()
            X = F.normalize(fea_cpu, p=2, dim=1).numpy()
            r, c = A.nonzero()
            sim = (X[r] * X[c]).sum(axis=1)
            sim = np.clip(sim, 0.0, 1.0)
            W = sp.csr_matrix((sim, (r, c)), shape=(n, n), dtype=np.float32)
        else:
            W = A.copy()

        d = np.asarray(W.sum(axis=1)).reshape(-1)
        iso = d < eps
        if iso.any():
            W = W.tolil()
            for u in np.where(iso)[0]:
                W[u, u] = 1.0
            W = W.tocsr()
            d = np.asarray(W.sum(axis=1)).reshape(-1)

        Dinv = sp.diags(1.0 / np.maximum(d, eps), format="csr")
        P = Dinv @ W

        v = np.zeros((n,), dtype=np.float32)
        v[seed_u] = 1.0
        x = alpha * v.copy()
        s = x.copy()
        for _ in range(1, N):
            x = (1.0 - alpha) * (P @ x)
            s += x

        k = max(1, int(round(topk_ratio * n)))
        idx = np.argpartition(s, -k)[-k:]
        topk_idx = idx[np.argsort(-s[idx])]
        return topk_idx, s

    def select_text_candidate_nodes(
        self,
        fea: torch.Tensor,
        adj: sp.spmatrix,
        row_idx: int,
        col_idx: int,
    ):
        k = self._text_candidate_budget(adj.shape[0])
        strategy = self.local_candidate_strategy

        if strategy == "local_ae_ppr":
            topk_idx, _ = self.ppr_topk_from_seeds(
                fea=fea,
                adj=adj,
                seed_nodes=[row_idx, col_idx],
                topk=k,
                alpha=self.text_ppr_alpha,
                N=self.text_ppr_iters,
                use_cos_sim=True,
            )
            return topk_idx

        if strategy == "random":
            pool = self.local_neighborhood_nodes(
                adj, [row_idx, col_idx], hops=self.local_candidate_hops
            )
            if pool.size <= k:
                return pool
            return np.sort(self.local_candidate_rng.choice(pool, size=k, replace=False))

        if strategy == "local_degree":
            pool = self.local_neighborhood_nodes(
                adj, [row_idx, col_idx], hops=self.local_candidate_hops
            )
            degrees = np.asarray(adj.tocsr().sum(axis=1)).reshape(-1)
            order = np.lexsort((pool, -degrees[pool]))
            return pool[order[:k]]

        if strategy == "global_pagerank":
            scores = self.compute_global_ppr(
                fea=fea,
                adj=adj,
                topk_ratio=self.text_topk_ratio,
                alpha=self.global_ppr_alpha,
                N=self.global_ppr_iters,
                use_cos_sim=False,
                seed_strategy="uniform",
            )
            idx = np.argpartition(scores, -k)[-k:]
            return idx[np.argsort(-scores[idx])]

        raise ValueError(f"Unsupported local_candidate_strategy: {strategy}")

    def _text_candidate_budget(self, n_nodes: int):
        if self.text_attack_nodes is not None:
            return min(n_nodes, self.text_attack_nodes)
        return max(1, int(round(self.text_topk_ratio * n_nodes)))

    def local_neighborhood_nodes(self, adj: sp.spmatrix, seed_nodes, hops: int = 2):
        A = adj.tocsr().astype(np.float32)
        A.setdiag(0)
        A.eliminate_zeros()

        seeds = [int(node) for node in seed_nodes]
        seen = set(seeds)
        frontier = np.asarray(seeds, dtype=np.int64)

        for _ in range(max(0, hops)):
            if frontier.size == 0:
                break
            neighbors = np.unique(A[frontier].nonzero()[1]).astype(np.int64)
            next_frontier = np.asarray(
                [int(node) for node in neighbors.tolist() if int(node) not in seen],
                dtype=np.int64,
            )
            if next_frontier.size == 0:
                break
            seen.update(next_frontier.tolist())
            frontier = next_frontier

        return np.asarray(sorted(seen), dtype=np.int64)

    def ppr_topk_from_seeds(
        self,
        fea: torch.Tensor,
        adj: sp.spmatrix,
        seed_nodes,
        topk: int,
        alpha: float = 0.2,
        N: int = 25,
        eps: float = 1e-12,
        use_cos_sim: bool = True,
    ):
        n = adj.shape[0]
        A = adj.tocsr().astype(np.float32)
        A.setdiag(0)
        A.eliminate_zeros()

        if use_cos_sim:
            fea_cpu = fea.detach().to("cpu").float()
            if fea_cpu.is_sparse:
                fea_cpu = fea_cpu.to_dense()
            X = F.normalize(fea_cpu, p=2, dim=1).numpy()
            r, c = A.nonzero()
            sim = (X[r] * X[c]).sum(axis=1)
            sim = np.clip(sim, 0.0, 1.0)
            W = sp.csr_matrix((sim, (r, c)), shape=(n, n), dtype=np.float32)
        else:
            W = A.copy()

        d = np.asarray(W.sum(axis=1)).reshape(-1)
        iso = d < eps
        if iso.any():
            W = W.tolil()
            for u in np.where(iso)[0]:
                W[u, u] = 1.0
            W = W.tocsr()
            d = np.asarray(W.sum(axis=1)).reshape(-1)

        Dinv = sp.diags(1.0 / np.maximum(d, eps), format="csr")
        P = Dinv @ W

        v = np.zeros((n,), dtype=np.float32)
        valid_seeds = [int(node) for node in seed_nodes if 0 <= int(node) < n]
        if not valid_seeds:
            valid_seeds = [0]
        for seed in valid_seeds:
            v[seed] += 1.0 / len(valid_seeds)

        s = v.copy()
        for _ in range(max(1, N)):
            s = alpha * v + (1.0 - alpha) * (P.T @ s)

        k = min(n, max(1, int(topk)))
        idx = np.argpartition(s, -k)[-k:]
        topk_idx = idx[np.argsort(-s[idx])]
        return topk_idx, s

    def compute_global_ppr(
        self,
        fea: torch.Tensor,
        adj: sp.spmatrix,
        topk_ratio: float = 0.10,
        alpha: float = 0.15,
        N: int = 30,
        eps: float = 1e-12,
        use_cos_sim: bool = True,
        seed_strategy: str = "uniform",  # "uniform", "degree", "label"
    ):
        """
        计算全局属性增强 PageRank，识别重要节点

        参数:
        - fea: [n, d] 节点特征
        - adj: scipy.sparse (n, n) 邻接矩阵
        - topk_ratio: 不使用（为兼容保留）
        - alpha: 重启概率 (0.15 更全局，0.85 更局部)
        - N: 迭代次数
        - use_cos_sim: 是否使用特征相似度加权边
        - seed_strategy: 初始分布策略
            - "uniform": 均匀分布（无偏全局）
            - "degree": 按度数加权（偏向枢纽）
            - "label": 按标签节点加权（偏向已知重要节点）

        返回:
        - scores: np.ndarray[float], shape (n,) - 全局 PPR 重要性分数
        """
        n = adj.shape[0]
        A = adj.tocsr().astype(np.float32)
        A.setdiag(0)
        A.eliminate_zeros()

        # 构建属性增强的转移矩阵
        if use_cos_sim:
            fea_cpu = fea.detach().to("cpu").float()
            # 处理稀疏张量：转为稠密
            if fea_cpu.is_sparse:
                fea_cpu = fea_cpu.to_dense()
            X = F.normalize(fea_cpu, p=2, dim=1).numpy()
            r, c = A.nonzero()
            sim = (X[r] * X[c]).sum(axis=1)
            sim = np.clip(sim, 0.0, 1.0)
            W = sp.csr_matrix((sim, (r, c)), shape=(n, n), dtype=np.float32)
        else:
            W = A.copy()

        # 处理孤立节点
        d = np.asarray(W.sum(axis=1)).reshape(-1)
        iso = d < eps
        if iso.any():
            W = W.tolil()
            for u in np.where(iso)[0]:
                W[u, u] = 1.0
            W = W.tocsr()
            d = np.asarray(W.sum(axis=1)).reshape(-1)

        # 行归一化：转移矩阵
        Dinv = sp.diags(1.0 / np.maximum(d, eps), format="csr")
        P = Dinv @ W

        # 初始分布 v（根据策略）
        v = np.ones((n,), dtype=np.float32) / n  # 默认均匀

        if seed_strategy == "degree":
            # 按度数加权（度数高的节点初始概率大）
            degrees = np.asarray(A.sum(axis=1)).reshape(-1)
            if degrees.sum() > 0:
                v = degrees / degrees.sum()
        elif seed_strategy == "label":
            # 如果有标签信息，可以从已知节点出发
            # 这里暂时用均匀分布，您可以传入 idx_train 来定制
            pass

        # PageRank with restart
        x = alpha * v.copy()
        s = x.copy()
        for _ in range(1, N):
            x = (1.0 - alpha) * (P @ x) + alpha * v
            s += x

        # 归一化分数到 [0, 1]
        s = s / (s.sum() + eps)

        return s
