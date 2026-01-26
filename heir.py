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
from gb_division_simple import GBCluster

# 文本攻击生成器导入
try:
    from text_attack_generator import TextAttackGenerator

    TEXT_ATTACK_AVAILABLE = True
except ImportError:
    TEXT_ATTACK_AVAILABLE = False
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
        self.gb_cluster = GBCluster(n_clusters=self.M, mode="euclidean", verbose=0)
        self.use_oracle = use_oracle

        # ========== 文本攻击生成器初始化 ==========
        self.use_text_attack = getattr(args, "use_text_attack", False)
        if self.use_text_attack and TEXT_ATTACK_AVAILABLE:
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
                    num_retries=getattr(args, "text_retries", 1),  # 默认只重试1次
                )
                print(
                    f"✅ Text attack generator initialized with {llm_type.upper()} API"
                )
        else:
            self.text_generator = None
            if self.use_text_attack:
                print(
                    "⚠️ Text attack requested but not available. Falling back to gradient-based attack."
                )

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

    def attack_features_with_text(
        self, target_nodes, full_features, labels_st, budget_per_node=20
    ):
        """
        使用文本生成方法攻击节点特征

        Args:
            target_nodes: 要攻击的节点索引列表
            full_features: 当前完整特征矩阵 [nnodes, nfeat]
            labels_st: 自训练标签
            budget_per_node: 每个节点的修改预算（词数）

        Returns:
            modified_features: 修改后的特征矩阵
        """
        if not self.use_text_attack or self.text_generator is None:
            print("⚠️ Text attack not available, skipping feature attack")
            return full_features

        modified_features = full_features.clone()
        success_count = 0
        fail_count = 0

        import time
        from tqdm import tqdm

        start_time = time.time()

        # 使用tqdm进度条
        pbar = tqdm(
            target_nodes,
            desc=f"🔥 LLM Text Attack",
            unit="node",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        # 批量生成对抗性文本
        for node_id in pbar:
            # 获取目标标签（翻转预测）
            pred_label = labels_st[node_id].item()
            target_label = (pred_label + 1) % self.nclass  # 简单翻转到下一个类别

            # 获取当前节点的特征向量
            current_bow = full_features[node_id].detach().cpu()

            # 转换为numpy
            if current_bow.is_sparse:
                current_bow = current_bow.to_dense().numpy()
            else:
                current_bow = current_bow.numpy()

            # 从BoW向量提取used_words和not_used_words（防止词表维度不匹配导致越界）
            try:
                vocab = self.text_generator.vocab
                vocab_size = len(vocab)
                idx_used = np.where(current_bow > 0)[0]
                idx_not = np.where(current_bow == 0)[0]
                # 只保留在词表范围内的索引
                idx_used = idx_used[idx_used < vocab_size]
                idx_not = idx_not[idx_not < vocab_size]
                used_words = [vocab[i] for i in idx_used]
                not_used_words = [vocab[i] for i in idx_not]

                # 限制词数（避免过长的列表）
                if len(used_words) > budget_per_node:
                    used_words = used_words[:budget_per_node]
                if len(not_used_words) > 100:  # 限制禁用词数量
                    not_used_words = not_used_words[:100]

                # 生成对抗性文本
                text = self.text_generator.generate_adversarial_text(
                    used_words=used_words, not_used_words=not_used_words, verbose=False
                )

                # 将文本转回BoW向量
                new_bow = self.text_generator.vectorizer.transform([text]).toarray()[0]
                # 与当前特征维度对齐（若词表维度与特征维度不一致则做安全对齐）
                feat_dim = modified_features.shape[1]
                if new_bow.shape[0] != feat_dim:
                    aligned = np.zeros(feat_dim, dtype=np.float32)
                    copy_len = min(new_bow.shape[0], feat_dim)
                    aligned[:copy_len] = new_bow[:copy_len]
                    new_bow = aligned

                # 更新特征
                modified_features[node_id] = (
                    torch.from_numpy(new_bow).float().to(self.device)
                )

                success_count += 1
                # 更新进度条显示
                pbar.set_postfix({"✓": success_count, "✗": fail_count})

            except Exception as e:
                fail_count += 1
                pbar.set_postfix({"✓": success_count, "✗": fail_count})
                if fail_count <= 3:  # 只显示前3个错误
                    tqdm.write(f"  ✗ Node {node_id} failed: {str(e)}")
                continue

        pbar.close()

        total_time = time.time() - start_time
        print(
            f"✅ Attack completed: {success_count} successful, {fail_count} failed in {total_time:.1f}s "
            f"({total_time/len(target_nodes):.1f}s/node)"
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
            self.full_adj = utils.sparse_mx_to_torch_sparse_tensor(full_adj_cpu).to(
                self.device
            )
            adj_norm = utils.normalize_adj_tensor(self.full_adj, sparse=True)
            self.inner_train(full_features, adj_norm, idx_train, idx_unlabeled, labels)
            embeddings = self.get_embeddings(full_features, adj_norm)

            # === 全局属性增强 PageRank：找出重要节点 ===
            # 计算全图的属性增强 PPR 分数
            global_ppr_scores = self.compute_global_ppr(
                fea=full_features,
                adj=full_adj_cpu,
                topk_ratio=0.10,  # 取前10%的重要节点
                alpha=0.15,
                N=30,
                use_cos_sim=True,
            )
            # 全局重要节点索引（按 PPR 分数排序）
            global_important_nodes = np.argsort(-global_ppr_scores)[
                : int(0.10 * self.nnodes)
            ]

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
                                full_features[nodes, :], dim=0
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
                    #    分别从两个 seed 出发调用 att_walk，得到各自 Top-k
                    #    设置 topk_ratio（如 0.05 表示前 5% 节点）
                    topk_nodes_1, scores_1 = self.ppr_topk_from_seed(
                        fea=full_features,
                        adj=full_adj_cpu,  # scipy.sparse csr_matrix
                        seed_u=row_idx,
                        topk_ratio=0.05,
                        alpha=0.2,
                        N=25,
                        use_cos_sim=True,
                    )

                    topk_nodes_2, scores_2 = self.ppr_topk_from_seed(
                        fea=full_features,
                        adj=full_adj_cpu,
                        seed_u=col_idx,
                        topk_ratio=0.05,
                        alpha=0.2,
                        N=25,
                        use_cos_sim=True,
                    )
                    # ===== 文本攻击：针对 PPR Top-k 并集 U 执行对抗性文本生成 =====
                    U = np.unique(np.concatenate([topk_nodes_1, topk_nodes_2]))

                    # 使用文本生成方法攻击 PPR 邻居节点 - 只攻击未攻击过的节点
                    if self.use_text_attack and self.text_generator is not None:
                        # 过滤掉已攻击过的节点
                        if not hasattr(self, "_attacked_nodes"):
                            self._attacked_nodes = set()

                        new_nodes = np.array(
                            [n for n in U if n not in self._attacked_nodes]
                        )

                        if len(new_nodes) > 0:
                            print(
                                f"\n🎯 [Step {tot_perturbs}/{n_perturbations}] "
                                f"Text attack: {len(new_nodes)} new nodes (skipped {len(U)-len(new_nodes)} already attacked)"
                            )

                            # 执行文本攻击 - 只攻击新节点
                            full_features = self.attack_features_with_text(
                                target_nodes=new_nodes,
                                full_features=full_features,
                                labels_st=labels_st,
                                budget_per_node=15,
                            )

                            # 记录已攻击的节点
                            self._attacked_nodes.update(new_nodes.tolist())
                        else:
                            print(
                                f"\n⏭️ [Step {tot_perturbs}/{n_perturbations}] "
                                f"All {len(U)} nodes already attacked, skipping"
                            )
                    else:
                        print(
                            f"⚠️ [Step {tot_perturbs}/{n_perturbations}] "
                            f"Text attack disabled, skipping {len(U)} nodes"
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
        # print(num_del, num_add, full_adj_cpu.sum(), 1.0 * depth / n_perturbations)
        if self.attack_structure:
            self.modified_adj = full_adj_cpu
        if self.attack_features:
            # 文本攻击已在循环中完成，直接保存最终特征
            self.modified_features = full_features.detach()
            print(
                f"\n✅ Attack completed: Structure perturbations={num_add+num_del}, Text attacks executed in-loop"
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
