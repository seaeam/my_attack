import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack

from fast_pytorch_kmeans import KMeans

class Heirattack(BaseAttack):
    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu', with_bias=False, lambda_=0.5, train_iters=10, lr=0.1, momentum=0.9, levels=2, use_oracle=False):

        super(Heirattack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        self.lambda_ = lambda_

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

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

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
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
        # self.kmeans = KMeans(n_clusters=self.M, mode='cosine', verbose=0)
        self.kmeans = KMeans(n_clusters=self.M, mode='euclidean', verbose=0)
        self.use_oracle = use_oracle

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def filter_potential_singletons(self, modified_adj):
        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def self_training_label(self, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        if self.use_oracle:
            return labels
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training

    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        t_d_min = torch.tensor(2.0).to(self.device)
        t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff)
        return allowed_mask, current_ratio

    def get_adj_score(self, adj_grad, is_add=True):
        adj_meta_grad = adj_grad * (1 if is_add else -1)
        return adj_meta_grad

    def get_feature_score(self, feature_grad, is_add=True):
        feature_meta_grad = feature_grad * (1 if is_add else -1)
        return feature_meta_grad

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

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

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

    def get_meta_grad(self, features, adj_norm, labels, labels_u):
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

        attack_loss = torch.sum(-output * labels) / (torch.sum(labels) + 1e-8)
        attack_loss = attack_loss + torch.sum(-output * labels_u) / (torch.sum(labels_u) + 1e-8)

        # print(attack_loss)

        # mask = torch.sum(labels, dim=1) > 0
        # attack_loss = F.nll_loss(output[mask], torch.argmax(labels[mask], dim=1))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.cur_adj, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad

    def meta_attack_multi_step(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, n_step=1, ll_constraint=True, ll_cutoff=0.004, type='Meta-Both'):
        self.sparse_features = sp.issparse(ori_features)
        self.sparse_adj = sp.issparse(ori_adj)

        full_adj, full_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        labels_st = self.self_training_label(labels, idx_train)
        labels_oh_l = torch.zeros(self.nnodes, self.nclass).to(self.device)
        labels_oh_ul = torch.zeros(self.nnodes, self.nclass).to(self.device)
        labels_oh_ul[idx_unlabeled] = labels_oh_ul[idx_unlabeled].scatter_(1, labels_st[idx_unlabeled].unsqueeze(1), 1)
        labels_oh_l[idx_train] = labels_oh_l[idx_train].scatter_(1, labels_st[idx_train].unsqueeze(1), 1)

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
            self.full_adj = utils.sparse_mx_to_torch_sparse_tensor(full_adj_cpu).to(self.device)
            adj_norm = utils.normalize_adj_tensor(self.full_adj, sparse=True)
            self.inner_train(full_features, adj_norm, idx_train, idx_unlabeled, labels)
            embeddings = self.get_embeddings(full_features, adj_norm)

            # KMeans
            pool = [range(self.nnodes)]
            childs = [[]]
            parents = [-1]
            levels = [1]
            cur = 0
            while cur < len(pool):
                subgraph = pool[cur]
                n = len(subgraph)

                if n <= self.M: # or levels[cur] >= self.levels: # Terminals
                    if len(subgraph) > 1:
                        for i in subgraph:
                            childs[cur].append(len(pool))
                            pool.append([i])
                            childs.append([])
                            parents.append(cur)
                            levels.append(levels[cur] + 1)
                    cur += 1
                    continue

                targets = []
                m = len(targets)
                if m >= self.M:
                    cid = self.kmeans.fit_predict(embeddings[targets, :])
                    cid = self.kmeans.fit_predict(embeddings[subgraph, :], centroids=self.kmeans.centroids)
                else:
                    cid = self.kmeans.fit_predict(embeddings[subgraph, :])

                for i in range(self.M):
                    childs[cur].append(len(pool))
                    pool.append([])
                    childs.append([])
                    parents.append(cur)
                    levels.append(levels[cur] + 1)
                for i in range(n):
                    id = cid[i].item()
                    # print(i, pool[cur][i], id, cid.max(), cid.min())
                    pool[-id-1].append(subgraph[i])

                cur += 1

            for step in range(n_step):
                tot_perturbs += 1
                if tot_perturbs > n_perturbations:
                    break

                while len((full_adj_cpu - ori_adj).nonzero()[0]) < tot_perturbs * 2:
                    # print((full_adj_cpu - ori_adj).nonzero()[0].shape)
                    # Root = 0
                    inpool_set = set(childs[0])
                    status = 'unknown'
                    targetI, targetJ = 0, 0

                    # while status == 'unknown' or len(childs[targetI]) > 0 or len(childs[targetJ]) > 0:
                    while True: # for level in range(self.levels):
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
                            feature_inpool[i] = torch.mean(full_features[nodes, :], dim=0)
                            # labels_inpool[i] = torch.sum(labels_onehot[nodes, :], dim=0)
                            if 'Both' in type or 'Train' in type:
                                labels_inpool_l[i] = torch.sum(labels_oh_l[nodes, :], dim=0)
                            if 'Both' in type or 'Self' in type:
                                labels_inpool_ul[i] = torch.sum(labels_oh_ul[nodes, :], dim=0)
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
                        adj_ip_norm = utils.normalize_adj_tensor(self.cur_adj).to(self.device)

                        adj_grad, feature_grad = self.get_meta_grad(feature_inpool, adj_ip_norm,
                                                                    labels_inpool_l, labels_inpool_ul)

                        posI, posJ = [], []
                        for i in range(n):
                            ip = inpool[i]
                            if parents[ip] == targetI or (ip == targetI and sizes[i] == 1):
                                posI.append(i)
                        for i in range(n):
                            ip = inpool[i]
                            if parents[ip] == targetJ or (ip == targetJ and sizes[i] == 1):
                                posJ.append(i)
                        best_score, best_status, I, J = -1e10, 'unknown', -1, -1
                        # best_score, best_status, I, J = -1e10, 'unknown', -1, -1

                        for i in posI:
                            for j in posJ:
                                if self.attack_structure:
                                    if status in ['unknown', 'add']:
                                        possible_edges = (sizes[i] - 1) * sizes[i] / 2 \
                                            if i == j else sizes[i] * sizes[j]
                                        possible_edges = possible_edges - deled_inpool[i][j]
                                        if possible_edges > adj_inpool[i][j]:
                                            score = adj_grad[i][j] # / (possible_edges - adj_inpool[i][j])
                                            if score > best_score:
                                                best_score, best_status, I, J = score, 'add', i, j
                                    if status in ['unknown', 'del']:
                                        if adj_inpool[i][j] > added_inpool[i][j]:
                                            score = -adj_grad[i][j] # / adj_inpool[i][j]
                                            if score > best_score:
                                                best_score, best_status, I, J = score, 'del', i, j

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


                    row_idx, col_idx = pool[targetI][0], pool[targetJ][0]
                    full_adj_cpu[row_idx, col_idx] = 1 - full_adj_cpu[row_idx, col_idx]
                    full_adj_cpu[col_idx, row_idx] = 1 - full_adj_cpu[col_idx, row_idx]
                    # print(full_adj_cpu.shape, row_idx, col_idx, full_adj_cpu[row_idx, col_idx])

                    if status == 'add':
                        num_add += 1
                        added[row_idx, col_idx] = 1
                        added[col_idx, row_idx] = 1
                    else:
                        num_del += 1
                        deled[row_idx, col_idx] = 1
                        deled[col_idx, row_idx] = 1

        print(num_del, num_add, full_adj_cpu.sum(), 1.0 * depth / n_perturbations)
        if self.attack_structure:
            self.modified_adj = full_adj_cpu
        if self.attack_features:
            self.modified_features = full_features.detach()

    def split_node(self, idx, embeddings, pool, parents, inpool):
        subgraph = pool[idx]
        n = len(subgraph)
        if n <= self.M:
            for i in subgraph:
                inpool.add(len(pool))
                pool.append([i])
                parents.append(idx)
            inpool.remove(idx)
            return pool, parents, inpool
        cid = self.kmeans.fit_predict(embeddings[subgraph, :])
        for i in range(self.M):
            inpool.add(len(pool))
            pool.append([])
            parents.append(idx)
        for i in range(n):
            id = cid[i].item()
            pool[-id - 1].append(subgraph[i])
        inpool.remove(idx)
        return pool, parents, inpool

    def meta_attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, n_step=1, ll_constraint=True, ll_cutoff=0.004, type='Meta-Both'):
        n_step = 1

        self.sparse_features = sp.issparse(ori_features)
        self.sparse_adj = sp.issparse(ori_adj)

        full_adj, full_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        labels_st = self.self_training_label(labels, idx_train)
        labels_oh_l = torch.zeros(self.nnodes, self.nclass).to(self.device)
        labels_oh_ul = torch.zeros(self.nnodes, self.nclass).to(self.device)
        labels_oh_ul[idx_unlabeled] = labels_oh_ul[idx_unlabeled].scatter_(1, labels_st[idx_unlabeled].unsqueeze(1), 1)
        labels_oh_l[idx_train] = labels_oh_l[idx_train].scatter_(1, labels_st[idx_train].unsqueeze(1), 1)


        n_turns = math.ceil(n_perturbations * 1.0 / n_step)
        tot_perturbs = 0
        full_adj_cpu = ori_adj.copy()
        added, deled = sp.csr_matrix(ori_adj.shape), sp.csr_matrix(ori_adj.shape)

        num_add, num_del, depth, ps = 0, 0, 0, 0

        for i in tqdm(range(n_turns), desc="Perturbing graph"):
            self.full_adj = utils.sparse_mx_to_torch_sparse_tensor(full_adj_cpu).to(self.device)
            adj_norm = utils.normalize_adj_tensor(self.full_adj, sparse=True)
            self.inner_train(full_features, adj_norm, idx_train, idx_unlabeled, labels)
            embeddings = self.get_embeddings(full_features, adj_norm).clone().detach()

            tot_perturbs += 1
            if tot_perturbs > n_perturbations:
                break

            pool, parents, inpool_set = [range(self.nnodes)], [-1], set([0])
            pool, parents, inpool_set = self.split_node(0, embeddings, pool, parents, inpool_set)

            status = 'unknown'
            targetI, targetJ = 0, 0

            while True:
                depth += 1
                ps += len(inpool_set)

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
                    feature_inpool[i] = torch.mean(full_features[nodes, :], dim=0)
                    if 'Both' in type or 'Train' in type:
                        labels_inpool_l[i] = torch.sum(labels_oh_l[nodes, :], dim=0)
                    if 'Both' in type or 'Self' in type:
                        labels_inpool_ul[i] = torch.sum(labels_oh_ul[nodes, :], dim=0)
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
                adj_ip_norm = utils.normalize_adj_tensor(self.cur_adj).to(self.device)

                adj_grad, feature_grad = self.get_meta_grad(feature_inpool, adj_ip_norm,
                                                            labels_inpool_l, labels_inpool_ul)

                posI, posJ = [], []
                for i in range(n):
                    ip = inpool[i]
                    if parents[ip] == targetI or (ip == targetI and sizes[i] == 1):
                        posI.append(i)
                for i in range(n):
                    ip = inpool[i]
                    if parents[ip] == targetJ or (ip == targetJ and sizes[i] == 1):
                        posJ.append(i)
                best_score, best_status, I, J = -1e10, 'unknown', -1, -1

                for i in posI:
                    for j in posJ:
                        if self.attack_structure:
                            if status in ['unknown', 'add']:
                                possible_edges = (sizes[i] - 1) * sizes[i] / 2 \
                                    if i == j else sizes[i] * sizes[j]
                                possible_edges = possible_edges - deled_inpool[i][j]
                                if possible_edges > adj_inpool[i][j]:
                                    score = adj_grad[i][j] # / (possible_edges - adj_inpool[i][j])
                                    if score > best_score:
                                        best_score, best_status, I, J = score, 'add', i, j
                            if status in ['unknown', 'del']:
                                if adj_inpool[i][j] > added_inpool[i][j]:
                                    score = -adj_grad[i][j] # / adj_inpool[i][j]
                                    if score > best_score:
                                        best_score, best_status, I, J = score, 'del', i, j

                status, targetI, targetJ = best_status, inpool[I], inpool[J]
                if sizes[I] == 1 and sizes[J] == 1:
                    break
                if sizes[I] > 1:
                    pool, parents, inpool_set = self.split_node(inpool[I], embeddings, pool, parents, inpool_set)
                if sizes[J] > 1 and not I == J:
                    pool, parents, inpool_set = self.split_node(inpool[J], embeddings, pool, parents, inpool_set)
                # print(inpool[I], inpool[J], inpool_set)

            row_idx, col_idx = pool[targetI][0], pool[targetJ][0]
            # print(status, n, row_idx in idx_unlabeled, col_idx in idx_unlabeled, row_idx, col_idx, full_adj_cpu[row_idx, col_idx])
            full_adj_cpu[row_idx, col_idx] = 1 - full_adj_cpu[row_idx, col_idx]
            full_adj_cpu[col_idx, row_idx] = 1 - full_adj_cpu[col_idx, row_idx]
            # print(full_adj_cpu.shape, row_idx, col_idx, full_adj_cpu[row_idx, col_idx])

            if status == 'add':
                num_add += 1
                added[row_idx, col_idx] = 1
                added[col_idx, row_idx] = 1
            else:
                num_del += 1
                deled[row_idx, col_idx] = 1
                deled[col_idx, row_idx] = 1


        print(num_del, num_add, full_adj_cpu.sum(), 1.0 * depth / n_perturbations, 1.0 * ps / depth)
        if self.attack_structure:
            self.modified_adj = full_adj_cpu
        if self.attack_features:
            self.modified_features = full_features.detach()
