import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from gb_division import gb_division
from deeprobust.graph.global_attack import BaseAttack
from fast_pytorch_kmeans import KMeans


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
        self.kmeans = KMeans(n_clusters=self.M, mode="euclidean", verbose=0)
        self.use_oracle = use_oracle

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

    # 防止产生孤立节点
    def filter_potential_singletons(self, modified_adj):
        degrees = modified_adj.sum(0)
        degree_one = degrees == 1
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    # 为未标记节点生成伪标签，用于自训练
    def self_training_label(self, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        if self.use_oracle:
            return labels
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training

    # 对每一条候选扰动边进行幂律分布约束筛选，防止攻击后图的度分布偏离原始幂律分布太多，以提升攻击的隐蔽性和合理性。
    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        t_d_min = torch.tensor(2.0).to(self.device)
        t_possible_edges = np.array(
            np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()
        ).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(
            t_possible_edges, modified_adj, ori_adj, t_d_min, ll_cutoff
        )
        return allowed_mask, current_ratio

    def get_adj_score(self, adj_grad, is_add=True):
        adj_meta_grad = adj_grad * (1 if is_add else -1)
        return adj_meta_grad

    def get_feature_score(self, feature_grad, is_add=True):
        feature_meta_grad = feature_grad * (1 if is_add else -1)
        return feature_meta_grad

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
        # 步骤1: 数据格式检查与转换
        self.sparse_features = sp.issparse(ori_features)
        self.sparse_adj = sp.issparse(ori_adj)

        full_adj, full_features, labels = utils.to_tensor(
            ori_adj, ori_features, labels, device=self.device
        )

        # 步骤2: 自训练标签生成 - 为未标记节点生成伪标签
        labels_st = self.self_training_label(labels, idx_train)
        labels_oh_l = torch.zeros(self.nnodes, self.nclass).to(self.device)
        labels_oh_ul = torch.zeros(self.nnodes, self.nclass).to(self.device)
        labels_oh_ul[idx_unlabeled] = labels_oh_ul[idx_unlabeled].scatter_(
            1, labels_st[idx_unlabeled].unsqueeze(1), 1
        )
        labels_oh_l[idx_train] = labels_oh_l[idx_train].scatter_(
            1, labels_st[idx_train].unsqueeze(1), 1
        )

        # 步骤3: 攻击参数初始化

        n_turns = math.ceil(n_perturbations * 1.0 / n_step)
        tot_perturbs = 0
        full_adj_cpu = ori_adj.copy()  # 当前扰动后的邻接矩阵
        added, deled = sp.csr_matrix(ori_adj.shape), sp.csr_matrix(
            ori_adj.shape
        )  # 记录已加/删的边

        num_add, num_del, depth = 0, 0, 0

        # new_data = gb_division(self.gb_data, self.args)
        # new_features = torch.from_numpy(new_data["gb_features"])
        self._ensure_gb_cache()

        # 步骤4: 主攻击循环 - 每轮执行一次或多次扰动
        for i in tqdm(range(n_turns), desc="Perturbing graph"):
            # 步骤4.1: 准备当前轮的图数据
            self.full_adj = utils.sparse_mx_to_torch_sparse_tensor(full_adj_cpu).to(
                self.device
            )
            adj_norm = utils.normalize_adj_tensor(self.full_adj, sparse=True)

            # 步骤4.2: 训练代理分类器 - 用当前扰动下的图训练轻量级代理模型
            self.inner_train(full_features, adj_norm, idx_train, idx_unlabeled, labels)

            # 步骤4.3: 获取节点嵌入 - 用于后续聚类分组
            embeddings = self.get_embeddings(full_features, adj_norm)

            # 步骤4.4: 构建层次化聚类结构 - KMeans递归分组
            pool = [range(self.nnodes)]  # 节点分组池
            childs = [[]]  # 子分组关系
            parents = [-1]  # 父分组关系
            levels = [1]  # 分组层级
            cur = 0
            while cur < len(pool):
                subgraph = pool[cur]
                n = len(subgraph)

                # 如果当前分组足够小，直接分解为单节点
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

                # TODO
                # 否则用KMeans聚类进一步细分
                # === A+B+C: Gated Semi Warm-Start for KMeans (drop-in replacement) ===
                # subgraph: 当前要划分的节点列表
                sub_np = np.asarray(list(subgraph), dtype=np.int64)
                emb_sub = embeddings[sub_np, :]  # [n, h] on self.device
                n = emb_sub.size(0)
                M = self.M  # 目标聚类数（与 self.kmeans 一致）

                # C) 度量对齐：若用 cosine，就对数据与质心做 L2 归一
                use_cosine = getattr(self.kmeans, "mode", "euclidean") == "cosine"
                if use_cosine:
                    emb_sub = F.normalize(emb_sub, p=2, dim=1)

                # 极端保护：n <= M 时，本来就不会走到这里；若走到，也给出退化处理
                if n <= M:
                    cid = self.kmeans.fit_predict(emb_sub)  # 退化回原版
                else:
                    # A) 门控：覆盖度、粒球多样性、样本量充足 才启用 warm-start
                    gb_ids = torch.from_numpy(self.node2gb[sub_np]).to(
                        self.device
                    )  # [n], -1 表示无映射
                    mask = gb_ids >= 0
                    coverage = mask.float().mean().item()  # 子图内“有粒球映射”的比例
                    uniq_gb = torch.unique(gb_ids[mask]).numel()  # 子图内粒球种类数

                    cov_thr = 0.50  # 原来 0.60 -> 0.50
                    uniq_thr = max(2, max(3, M // 4))  # 原来 M//2 -> M//4，且至少 3
                    n_thr = int(M + max(5, 0.25 * M))  # 原来 2M -> M + max(5, 0.25M)
                    USE_WARM = (
                        (coverage >= cov_thr) and (uniq_gb >= uniq_thr) and (n >= n_thr)
                    )

                    if not USE_WARM:
                        # 回退：用原始 KMeans 初始化
                        cid = self.kmeans.fit_predict(emb_sub)
                    else:
                        # B) 半暖启动：一半质心来自“粒球均值”，一半用最远点采样（kmeans++ 近似）
                        # 先取 Top-k_gb 个粒球（按在 subgraph 中出现频次）
                        uniq, counts = torch.unique(
                            gb_ids[mask], return_counts=True
                        )  # 只统计有效粒球
                        order = torch.argsort(counts, descending=True)
                        k_gb = int(
                            max(1, min(uniq.numel(), math.floor(0.5 * M)))
                        )  # ≈ 50% 用粒球初始化

                        init_c_list = []
                        taken = 0
                        for g in uniq[order]:
                            if taken >= k_gb:
                                break
                            sel = gb_ids == g
                            if sel.any():
                                init_c_list.append(
                                    emb_sub[sel].mean(0, keepdim=True)
                                )  # 该粒球在 subgraph 内的均值嵌入
                                taken += 1

                        # 边界情况：若没有有效粒球，直接回退
                        if len(init_c_list) == 0:
                            cid = self.kmeans.fit_predict(emb_sub)
                        else:
                            init_centroids = torch.cat(init_c_list, dim=0)  # [k_gb, h]

                            # 最远点采样（kmeans++ 近似）补足剩余质心
                            def farthest_pick(
                                X: torch.Tensor, C: torch.Tensor, need: int
                            ) -> torch.Tensor:
                                # X: [n, h], C: [m, h]
                                if C.numel() == 0:
                                    j = torch.randint(
                                        0, X.size(0), (1,), device=X.device
                                    ).item()
                                    C = X[j : j + 1]
                                    need -= 1
                                for _ in range(max(0, need)):
                                    # 到最近质心的距离平方
                                    d2 = torch.cdist(X, C).pow(2).min(dim=1).values
                                    probs = (d2 / (d2.sum() + 1e-12)).clamp(min=1e-12)
                                    j = torch.multinomial(probs, 1).item()
                                    C = torch.cat([C, X[j : j + 1]], dim=0)
                                return C

                            init_centroids = farthest_pick(
                                emb_sub, init_centroids, M - init_centroids.size(0)
                            )
                            if use_cosine:
                                init_centroids = F.normalize(init_centroids, p=2, dim=1)

                            # 最终用 warm-start 质心跑 KMeans
                            cid = self.kmeans.fit_predict(
                                emb_sub, centroids=init_centroids
                            )
                # === end of A+B+C block ===

                # 创建M个子分组
                for i in range(self.M):
                    childs[cur].append(len(pool))
                    pool.append([])
                    childs.append([])
                    parents.append(cur)
                    levels.append(levels[cur] + 1)

                # 将节点分配到对应的子分组
                for i in range(n):
                    id = cid[i].item()
                    # print(i, pool[cur][i], id, cid.max(), cid.min())
                    pool[-id - 1].append(subgraph[i])

                cur += 1

            # 步骤4.5: 执行n_step次扰动
            for step in range(n_step):
                tot_perturbs += 1
                if tot_perturbs > n_perturbations:
                    break

                # 步骤4.5.1: 递归选择最优扰动位置
                while len((full_adj_cpu - ori_adj).nonzero()[0]) < tot_perturbs * 2:
                    # print((full_adj_cpu - ori_adj).nonzero()[0].shape)
                    # 从根节点开始递归搜索
                    inpool_set = set(childs[0])
                    status = "unknown"
                    targetI, targetJ = 0, 0

                    # 递归细化到单节点级别
                    while True:  # for level in range(self.levels):
                        depth += 1

                        # 步骤4.5.2: 构建当前层的粗化图
                        inpool = list(inpool_set)
                        n = len(inpool)
                        adj_inpool = np.zeros((n, n))  # 粗化图邻接矩阵
                        added_inpool = np.zeros((n, n))  # 粗化图已加边矩阵
                        deled_inpool = np.zeros((n, n))  # 粗化图已删边矩阵
                        feature_inpool = torch.zeros((n, self.nfeat)).to(
                            self.device
                        )  # 粗化图特征
                        labels_inpool_l = torch.zeros((n, self.nclass)).to(
                            self.device
                        )  # 训练集标签
                        labels_inpool_ul = torch.zeros((n, self.nclass)).to(
                            self.device
                        )  # 未标记集标签

                        sizes = torch.zeros((n, 1)).to(
                            self.device
                        )  # 每个粗化节点包含的原始节点数
                        cids = np.zeros(self.nnodes)  # 原始节点到粗化节点的映射

                        # 步骤4.5.3: 聚合粗化节点的特征、标签、大小
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

                        # 步骤4.5.4: 构建粗化图的邻接关系
                        for row, col in zip(*full_adj_cpu.nonzero()):
                            i, j = int(cids[row]), int(cids[col])
                            adj_inpool[i][j] = adj_inpool[i][j] + 1
                        for row, col in zip(*added.nonzero()):
                            i, j = int(cids[row]), int(cids[col])
                            added_inpool[i][j] = added_inpool[i][j] + 1
                        for row, col in zip(*deled.nonzero()):
                            i, j = int(cids[row]), int(cids[col])
                            deled_inpool[i][j] = deled_inpool[i][j] + 1

                        # 步骤4.5.5: 设置粗化图邻接矩阵并计算梯度
                        self.cur_adj = torch.Tensor(adj_inpool).to(self.device)
                        for i in range(n):
                            self.cur_adj[i][i] = self.cur_adj[i][i] + sizes[i] - 1
                        self.cur_adj.requires_grad = True
                        adj_ip_norm = utils.normalize_adj_tensor(self.cur_adj).to(
                            self.device
                        )

                        # 计算攻击损失对粗化图的梯度
                        adj_grad, feature_grad = self.get_meta_grad(
                            feature_inpool,
                            adj_ip_norm,
                            labels_inpool_l,
                            labels_inpool_ul,
                        )

                        # 步骤4.5.6: 确定候选扰动的粗化节点对
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

                        # 步骤4.5.7: 选择最优扰动操作（加边或删边）
                        for i in posI:
                            for j in posJ:
                                if self.attack_structure:
                                    # 尝试加边操作
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
                                    # 尝试删边操作
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

                        # 步骤4.5.8: 更新目标节点并检查是否需要进一步细分
                        status, targetI, targetJ = best_status, inpool[I], inpool[J]
                        if sizes[I] == 1 and sizes[J] == 1:
                            break  # 已经细分到单节点，结束递归

                        # 继续细分尚未到单节点的分组
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
                    # ===== 属性攻击：只对 Top-k 并集 U 动手 =====
                    U = np.unique(np.concatenate([topk_nodes_1, topk_nodes_2]))

                    # 注意：要用当前这一轮的 adj_norm（你上面已计算）
                    # 可根据特征类型调整预算与参数：
                    full_features = self.attribute_step(
                        full_features=full_features,
                        adj_norm=adj_norm,
                        labels_oh_l=labels_oh_l,
                        labels_oh_ul=labels_oh_ul,
                        U=U,
                        feature_type="auto",  # "auto" 会自动判断二值/连续
                        B_feat=None,  # 二值时全局预算；None 时默认 0.1% 维度
                        b_node=10,  # 二值时每节点最多翻转 10 维，可按需调
                        eps=0.05,  # 连续特征 Linf 半径
                        steps=1,  # 1=FGSM；>1=PGD
                        step_size=None,  # None=默认等于 eps
                        clip_min=0.0,
                        clip_max=1.0,
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
            self.modified_features = full_features.detach()

    # 层次化聚类中的节点分割函数 - 将一个节点分组进一步细分为更小的子分组
    # 这是实现"分治"策略的核心函数，用于递归地将大的节点集合分解为小的管理单元
    def split_node(self, idx, embeddings, pool, parents, inpool):
        """
        将指定的节点分组细分为更小的子分组

        参数:
        - idx: 要分割的分组在pool中的索引
        - embeddings: 所有节点的嵌入表示 [nnodes, embedding_dim]
        - pool: 节点分组池，每个元素是一个节点列表
        - parents: 每个分组的父分组索引
        - inpool: 当前活跃分组的索引集合

        返回:
        - 更新后的 pool, parents, inpool
        """
        # 步骤1: 获取当前分组包含的节点
        subgraph = pool[idx]  # 当前分组包含的节点列表
        n = len(subgraph)  # 当前分组的节点数量

        # 步骤2: 判断是否需要进一步分割
        # 如果节点数小于等于M，直接分解为单节点分组
        if n <= self.M:
            # 情况1: 节点数较少，直接分解为单个节点
            for i in subgraph:
                inpool.add(len(pool))  # 新分组的索引
                pool.append([i])  # 创建只包含单个节点的分组
                parents.append(idx)  # 设置父分组为当前分组
            inpool.remove(idx)  # 从活跃分组中移除当前分组
            return pool, parents, inpool

        # 步骤3: 使用KMeans聚类进行分割
        # 情况2: 节点数较多，需要用聚类算法细分
        cid = self.kmeans.fit_predict(
            embeddings[subgraph, :]
        )  # 对当前分组的节点嵌入进行K均值聚类

        # 步骤4: 创建M个空的子分组
        for i in range(self.M):
            inpool.add(len(pool))  # 新子分组的索引
            pool.append([])  # 创建空的子分组
            parents.append(idx)  # 设置父分组为当前分组

        # 步骤5: 将节点按聚类结果分配到对应的子分组
        for i in range(n):
            id = cid[i].item()  # 获取第i个节点的聚类标签
            pool[-id - 1].append(subgraph[i])  # 将节点添加到对应的子分组中
            # 注意: pool[-id-1] 是因为我们刚创建了M个子分组，它们在pool的末尾
            # -id-1 将聚类标签映射到正确的子分组索引

        # 步骤6: 更新活跃分组集合
        inpool.remove(idx)  # 从活跃分组中移除已分割的分组

        return pool, parents, inpool

    # 层次化图对抗攻击的简化版元攻击函数
    # 与meta_attack_multi_step相比，这是一个更直接的单步攻击实现
    def meta_attack(
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
        """
        简化版的层次化元攻击函数

        参数:
        - ori_features: 原始节点特征矩阵
        - ori_adj: 原始邻接矩阵
        - labels: 节点标签
        - idx_train: 训练集节点索引
        - idx_unlabeled: 未标记节点索引
        - n_perturbations: 总扰动预算
        - n_step: 每轮扰动步数（固定为1）
        - ll_constraint: 是否启用幂律约束
        - ll_cutoff: 幂律约束阈值
        - type: 攻击类型（Meta-Both, Meta-Train, Meta-Self）
        """

        # 步骤1: 强制设置单步攻击模式
        n_step = 1  # 固定为单步攻击，每轮只做一次扰动

        # 步骤2: 数据格式检查与转换
        self.sparse_features = sp.issparse(ori_features)
        self.sparse_adj = sp.issparse(ori_adj)

        full_adj, full_features, labels = utils.to_tensor(
            ori_adj, ori_features, labels, device=self.device
        )

        # 步骤3: 生成自训练标签 - 为未标记节点生成伪标签
        labels_st = self.self_training_label(labels, idx_train)
        labels_oh_l = torch.zeros(self.nnodes, self.nclass).to(
            self.device
        )  # 训练集标签(one-hot)
        labels_oh_ul = torch.zeros(self.nnodes, self.nclass).to(
            self.device
        )  # 未标记集标签(one-hot)

        # 将伪标签转换为one-hot格式
        labels_oh_ul[idx_unlabeled] = labels_oh_ul[idx_unlabeled].scatter_(
            1, labels_st[idx_unlabeled].unsqueeze(1), 1
        )
        labels_oh_l[idx_train] = labels_oh_l[idx_train].scatter_(
            1, labels_st[idx_train].unsqueeze(1), 1
        )

        # 步骤4: 攻击参数初始化
        n_turns = math.ceil(n_perturbations * 1.0 / n_step)  # 攻击轮数
        tot_perturbs = 0  # 已完成扰动计数
        full_adj_cpu = ori_adj.copy()  # 当前邻接矩阵
        added, deled = sp.csr_matrix(ori_adj.shape), sp.csr_matrix(
            ori_adj.shape
        )  # 记录已加/删边

        num_add, num_del, depth, ps = (
            0,
            0,
            0,
            0,
        )  # 统计信息: 加边数、删边数、递归深度、池大小

        # 步骤5: 主攻击循环 - 每轮执行一次扰动
        for i in tqdm(range(n_turns), desc="Perturbing graph"):
            # 步骤5.1: 准备当前轮的图数据
            self.full_adj = utils.sparse_mx_to_torch_sparse_tensor(full_adj_cpu).to(
                self.device
            )
            adj_norm = utils.normalize_adj_tensor(self.full_adj, sparse=True)

            # 步骤5.2: 训练代理分类器并获取节点嵌入
            self.inner_train(full_features, adj_norm, idx_train, idx_unlabeled, labels)
            embeddings = self.get_embeddings(full_features, adj_norm).clone().detach()

            # 步骤5.3: 检查扰动预算
            tot_perturbs += 1
            if tot_perturbs > n_perturbations:
                break

            # 步骤5.4: 初始化分组结构并执行初始分割
            pool, parents, inpool_set = [range(self.nnodes)], [-1], set([0])
            # 直接对根分组(所有节点)调用split_node进行首次分割
            pool, parents, inpool_set = self.split_node(
                0, embeddings, pool, parents, inpool_set
            )

            # 步骤5.5: 初始化递归搜索状态
            status = "unknown"  # 攻击状态: "unknown", "add", "del"
            targetI, targetJ = 0, 0  # 目标分组索引

            # 步骤5.6: 递归搜索最优扰动位置
            while True:
                depth += 1  # 递归深度计数
                ps += len(inpool_set)  # 累计活跃分组数量

                # 步骤5.6.1: 构建当前层的粗化图
                inpool = list(inpool_set)  # 当前活跃分组列表
                n = len(inpool)  # 粗化图节点数

                # 初始化粗化图数据结构
                adj_inpool = np.zeros((n, n))  # 粗化图邻接矩阵
                added_inpool = np.zeros((n, n))  # 粗化图已加边矩阵
                deled_inpool = np.zeros((n, n))  # 粗化图已删边矩阵
                feature_inpool = torch.zeros((n, self.nfeat)).to(
                    self.device
                )  # 粗化图特征
                labels_inpool_l = torch.zeros((n, self.nclass)).to(
                    self.device
                )  # 粗化图训练标签
                labels_inpool_ul = torch.zeros((n, self.nclass)).to(
                    self.device
                )  # 粗化图未标记标签

                sizes = torch.zeros((n, 1)).to(
                    self.device
                )  # 每个粗化节点包含的原始节点数
                cids = np.zeros(self.nnodes)  # 原始节点到粗化节点的映射

                # 步骤5.6.2: 聚合粗化节点的特征、标签和大小信息
                for i in range(n):
                    nodes = pool[inpool[i]]  # 粗化节点包含的原始节点
                    sizes[i] = len(nodes)  # 粗化节点大小

                    # 聚合特征: 取原始节点特征的均值
                    feature_inpool[i] = torch.mean(full_features[nodes, :], dim=0)

                    # 根据攻击类型聚合标签
                    if "Both" in type or "Train" in type:
                        labels_inpool_l[i] = torch.sum(labels_oh_l[nodes, :], dim=0)
                    if "Both" in type or "Self" in type:
                        labels_inpool_ul[i] = torch.sum(labels_oh_ul[nodes, :], dim=0)

                    # 建立原始节点到粗化节点的映射
                    for j in nodes:
                        cids[j] = i

                # 步骤5.6.3: 构建粗化图的邻接关系
                # 统计原图中的边
                for row, col in zip(*full_adj_cpu.nonzero()):
                    i, j = int(cids[row]), int(cids[col])
                    adj_inpool[i][j] = adj_inpool[i][j] + 1

                # 统计已添加的边
                for row, col in zip(*added.nonzero()):
                    i, j = int(cids[row]), int(cids[col])
                    added_inpool[i][j] = added_inpool[i][j] + 1

                # 统计已删除的边
                for row, col in zip(*deled.nonzero()):
                    i, j = int(cids[row]), int(cids[col])
                    deled_inpool[i][j] = deled_inpool[i][j] + 1

                # 步骤5.6.4: 设置粗化图邻接矩阵并计算梯度
                self.cur_adj = torch.Tensor(adj_inpool).to(self.device)
                # 添加自环: 粗化节点内部的连接数
                for i in range(n):
                    self.cur_adj[i][i] = self.cur_adj[i][i] + sizes[i] - 1
                self.cur_adj.requires_grad = True
                adj_ip_norm = utils.normalize_adj_tensor(self.cur_adj).to(self.device)

                # 计算攻击损失对粗化图的梯度
                adj_grad, feature_grad = self.get_meta_grad(
                    feature_inpool, adj_ip_norm, labels_inpool_l, labels_inpool_ul
                )

                # 步骤5.6.5: 确定候选扰动的粗化节点对
                posI, posJ = [], []

                # 找到与目标分组I相关的候选粗化节点
                for i in range(n):
                    ip = inpool[i]
                    if parents[ip] == targetI or (ip == targetI and sizes[i] == 1):
                        posI.append(i)

                # 找到与目标分组J相关的候选粗化节点
                for i in range(n):
                    ip = inpool[i]
                    if parents[ip] == targetJ or (ip == targetJ and sizes[i] == 1):
                        posJ.append(i)

                best_score, best_status, I, J = -1e10, "unknown", -1, -1

                # 步骤5.6.6: 枚举所有候选节点对，选择最优扰动操作
                for i in posI:
                    for j in posJ:
                        if self.attack_structure:
                            # 尝试加边操作
                            if status in ["unknown", "add"]:
                                # 计算可能的边数
                                possible_edges = (
                                    (sizes[i] - 1) * sizes[i] / 2  # 自环: C(n,2)
                                    if i == j
                                    else sizes[i] * sizes[j]  # 两个不同分组间: n*m
                                )
                                possible_edges = (
                                    possible_edges - deled_inpool[i][j]
                                )  # 减去已删除的边

                                if possible_edges > adj_inpool[i][j]:  # 还有边可以添加
                                    score = adj_grad[i][j]  # 加边的梯度分数
                                    if score > best_score:
                                        best_score, best_status, I, J = (
                                            score,
                                            "add",
                                            i,
                                            j,
                                        )

                            # 尝试删边操作
                            if status in ["unknown", "del"]:
                                if (
                                    adj_inpool[i][j] > added_inpool[i][j]
                                ):  # 还有边可以删除
                                    score = -adj_grad[i][j]  # 删边的梯度分数(取负)
                                    if score > best_score:
                                        best_score, best_status, I, J = (
                                            score,
                                            "del",
                                            i,
                                            j,
                                        )

                # 步骤5.6.7: 更新目标分组并检查是否需要继续细分
                status, targetI, targetJ = best_status, inpool[I], inpool[J]

                # 如果两个目标分组都已经是单节点，结束递归
                if sizes[I] == 1 and sizes[J] == 1:
                    break

                # 继续细分尚未到单节点的分组
                if sizes[I] > 1:
                    pool, parents, inpool_set = self.split_node(
                        inpool[I], embeddings, pool, parents, inpool_set
                    )
                if sizes[J] > 1 and not I == J:  # 避免重复分割同一个分组
                    pool, parents, inpool_set = self.split_node(
                        inpool[J], embeddings, pool, parents, inpool_set
                    )

            # 步骤5.7: 执行实际的扰动操作
            row_idx, col_idx = pool[targetI][0], pool[targetJ][0]  # 获取目标节点对

            # 在邻接矩阵上执行加边/删边操作(异或操作: 0变1, 1变0)
            full_adj_cpu[row_idx, col_idx] = 1 - full_adj_cpu[row_idx, col_idx]
            full_adj_cpu[col_idx, row_idx] = 1 - full_adj_cpu[col_idx, row_idx]

            # 步骤5.8: 记录扰动统计信息
            if status == "add":
                num_add += 1
                added[row_idx, col_idx] = 1
                added[col_idx, row_idx] = 1
            else:
                num_del += 1
                deled[row_idx, col_idx] = 1
                deled[col_idx, row_idx] = 1

        # 步骤6: 输出攻击结果和统计信息
        print(
            num_del,  # 删边数
            num_add,  # 加边数
            full_adj_cpu.sum(),  # 总边数
            1.0 * depth / n_perturbations,  # 平均递归深度
            1.0 * ps / depth,  # 平均池大小
        )

        # 步骤7: 保存最终的攻击结果
        if self.attack_structure:
            self.modified_adj = full_adj_cpu
        if self.attack_features:
            self.modified_features = full_features.detach()

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

    def attribute_step(
        self,
        full_features: torch.Tensor,
        adj_norm: torch.Tensor,
        labels_oh_l: torch.Tensor,
        labels_oh_ul: torch.Tensor,
        U: np.ndarray,
        *,
        feature_type: str = "auto",  # "auto" | "binary" | "continuous"
        B_feat: int = None,  # 全局预算（二值：最多翻转多少个维度；连续：忽略）
        b_node: int = 10,  # 每节点预算（二值：每节点最多翻转多少维；连续：忽略）
        # 连续特征攻击参数
        eps: float = 0.05,  # Linf 半径
        steps: int = 1,  # 1=FGSM；>1 为 PGD
        step_size: float = None,  # 每步步长，默认等于 eps
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        """
        仅对 U 中节点做属性攻击。返回更新后的 full_features（新的张量）。
        """
        device = self.device
        U = np.asarray(U, dtype=np.int64)
        if U.size == 0:
            return full_features

        U_t = torch.from_numpy(U).long().to(device)

        # 判定特征类型
        if feature_type == "auto":
            with torch.no_grad():
                is_binary = torch.all(
                    (full_features == 0) | (full_features == 1)
                ).item()
            feature_type = "binary" if is_binary else "continuous"

        # 准备前向，拿到对输入特征的梯度
        X = full_features.detach().clone().to(device)
        X.requires_grad_(True)

        hidden = X
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            # 强制走稠密分支，确保能对特征求梯度
            hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)
        attack_loss = torch.sum(-output * labels_oh_l) / (torch.sum(labels_oh_l) + 1e-8)
        attack_loss += torch.sum(-output * labels_oh_ul) / (
            torch.sum(labels_oh_ul) + 1e-8
        )

        grad = torch.autograd.grad(attack_loss, X, retain_graph=False)[0]  # [n, d]
        grad_U = grad[U_t].detach()

        if feature_type == "binary":
            # 预算默认值：0.1% 的全局维度
            if B_feat is None:
                B_feat = max(1, int(0.001 * X.numel()))

            XU = X[U_t].detach().clone()
            pos_mask = XU == 0
            neg_mask = XU == 1

            pos_scores = torch.where(pos_mask, grad_U, torch.full_like(grad_U, -1e9))
            neg_scores = torch.where(
                neg_mask, -grad_U, torch.full_like(grad_U, -1e9)
            )  # 越大越该翻 1->0

            flips_idx_per_node = []
            total_flips = 0
            for i in range(len(U)):
                # 每节点一半给 0->1，剩余给 1->0（也可以都让贪心全局裁剪）
                k1 = min(b_node // 2, int(pos_mask[i].sum().item()))
                k2 = min(b_node - k1, int(neg_mask[i].sum().item()))

                cand1 = (
                    torch.topk(pos_scores[i], k1).indices
                    if k1 > 0
                    else torch.tensor([], dtype=torch.long, device=device)
                )
                cand1 = cand1[pos_scores[i][cand1] > -1e8]

                cand2 = (
                    torch.topk(neg_scores[i], k2).indices
                    if k2 > 0
                    else torch.tensor([], dtype=torch.long, device=device)
                )
                cand2 = cand2[neg_scores[i][cand2] > -1e8]

                cands = torch.unique(torch.cat([cand1, cand2], dim=0))
                flips_idx_per_node.append(cands)
                total_flips += int(cands.numel())

            # 全局预算裁剪
            keep = None
            if total_flips > B_feat:
                contrib_list = []
                for i, cands in enumerate(flips_idx_per_node):
                    if cands.numel() == 0:
                        continue
                    gi = grad_U[i][cands]
                    xi = XU[i][cands]
                    # 0->1 用 +grad；1->0 用 -grad
                    contrib = torch.where(xi == 0, gi, -gi)
                    for j, fj in zip(cands.tolist(), contrib.tolist()):
                        contrib_list.append((i, j, float(fj)))
                contrib_list.sort(key=lambda x: x[2], reverse=True)
                keep = set([(i, j) for i, j, _ in contrib_list[:B_feat]])

            # 应用翻转
            XU_new = XU.clone()
            for i, cands in enumerate(flips_idx_per_node):
                if cands.numel() == 0:
                    continue
                for j in cands.tolist():
                    if (keep is not None) and ((i, j) not in keep):
                        continue
                    XU_new[i, j] = 1.0 - XU_new[i, j]

            X = X.detach()
            X[U_t] = XU_new
            return X

        else:
            # 连续：FGSM/PGD
            if step_size is None:
                step_size = eps
            X_adv = X.detach().clone()
            XU_adv = X_adv[U_t].clone()
            base = X_adv[U_t].clone()

            g = grad_U.clone()
            for t in range(steps):
                XU_adv = XU_adv + step_size * torch.sign(g)
                # 投影到 Linf ball
                XU_adv = torch.max(torch.min(XU_adv, base + eps), base - eps)
                # 裁剪到合法范围
                XU_adv = torch.clamp(XU_adv, clip_min, clip_max)

                if steps > 1:
                    # 重算梯度（只对 U）
                    X_tmp = X_adv.clone()
                    X_tmp[U_t] = XU_adv
                    X_tmp.requires_grad_(True)
                    h = X_tmp
                    for ix, w in enumerate(self.weights):
                        b = self.biases[ix] if self.with_bias else 0
                        h = adj_norm @ h @ w + b
                        if self.with_relu and ix != len(self.weights) - 1:
                            h = F.relu(h)
                    out = F.log_softmax(h, dim=1)
                    loss = torch.sum(-out * labels_oh_l) / (
                        torch.sum(labels_oh_l) + 1e-8
                    )
                    loss += torch.sum(-out * labels_oh_ul) / (
                        torch.sum(labels_oh_ul) + 1e-8
                    )
                    g = torch.autograd.grad(loss, X_tmp, retain_graph=False)[0][
                        U_t
                    ].detach()

            X_adv[U_t] = XU_adv.detach()
            return X_adv

    def global_ppr_topk(
        self,
        fea: torch.Tensor,
        adj: sp.spmatrix,
        idx_seed: np.ndarray = None,  # 起点分布 v 的支持集；默认全图或攻击候选
        topk_ratio: float = 0.08,  # 全局 Top-k 比例，建议 5%–10%
        alpha: float = 0.25,  # 重启概率，0.15 更局部，0.25 更全局
        N: int = 35,  # 迭代步/近似阶数
        eps: float = 1e-12,
        use_cos_sim: bool = True,
        degree_debias: float = 0.0,  # 枢纽去偏，>0 则用 p[i] / deg(i)^beta
    ):
        """
        全局属性增强 PPR：
        - fea: [n, d] 节点特征（torch.Tensor，CPU/GPU 均可）
        - adj: scipy.sparse CSR 邻接（无向）
        - idx_seed: 作为起点分布 v 的节点集合，若 None 则默认全图或攻击候选
        返回:
        - topk_idx: np.ndarray[int]，Top-k 节点索引
        - scores: np.ndarray[float]，长度 n 的 PPR 分数（可用于解释/分配预算）
        """
        n = adj.shape[0]
        A = adj.tocsr().astype(np.float32)
        A.setdiag(0)
        A.eliminate_zeros()

        # 1) 属性增强的边权（仅在现有边上算相似度）
        if use_cos_sim:
            X = F.normalize(fea.detach().to("cpu").float(), p=2, dim=1).numpy()
            r, c = A.nonzero()
            sim = (X[r] * X[c]).sum(axis=1)
            sim = np.clip(sim, 0.0, 1.0)  # 负相似度置零（可更稳）
            W = sp.csr_matrix((sim, (r, c)), shape=(n, n), dtype=np.float32)
        else:
            W = A.copy()

        # 2) 行归一化得到转移矩阵 P
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

        # 3) 构造全局起点分布 v
        if idx_seed is None:
            # 默认用“攻击候选集合”（若你传 idx_attack 更好；否则全图均匀）
            idx_seed = np.arange(n, dtype=np.int64)
        else:
            idx_seed = np.asarray(idx_seed, dtype=np.int64)

        v = np.zeros((n,), dtype=np.float32)
        if idx_seed.size > 0:
            v[idx_seed] = 1.0 / float(idx_seed.size)
        else:
            v[:] = 1.0 / float(n)

        # 4) 用带重启的 power-iteration 累积有限项近似
        x = alpha * v.copy()
        s = x.copy()
        for _ in range(1, N):
            x = (1.0 - alpha) * (P @ x)
            s += x

        # 5) 可选：对枢纽做去偏（避免把预算砸在高度节点上）
        if degree_debias > 0.0:
            deg = np.asarray(A.sum(axis=1)).reshape(-1) + 1e-12
            s = s / (deg**degree_debias)

        # 6) 取 Top-k
        k = max(1, int(round(topk_ratio * n)))
        idx = np.argpartition(s, -k)[-k:]
        topk_idx = idx[np.argsort(-s[idx])]
        return topk_idx, s

    def _ensure_gb_cache(self):
        if getattr(self, "_gb_cached", False):
            return

        gb_f, _ = gb_division(self.gb_data, self.args, fea=None)

        # --- graceful fallback if keys missing ---
        if "gb2nodes" not in gb_f or "node2gb" not in gb_f:
            # 退化：不开启 warm-start，但不报错
            self.node2gb = np.full(self.nnodes, -1, dtype=np.int64)
            self.gb_features = torch.tensor(
                gb_f.get("gb_features", np.zeros((0, self.nfeat), dtype=np.float32)),
                dtype=self.features.dtype,
                device=self.device,
            )
            self._gb_cached = True
            return
        # ----------------------------------------

        # 正常路径：把映射补齐到 nnodes 长度
        self.node2gb = np.full(self.nnodes, -1, dtype=np.int64)
        for gid, nodes in enumerate(gb_f["gb2nodes"]):
            self.node2gb[np.asarray(nodes, dtype=np.int64)] = gid

        self.gb_features = torch.tensor(
            gb_f["gb_features"], dtype=self.features.dtype, device=self.device
        )
        self._gb_cached = True
