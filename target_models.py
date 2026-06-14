from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, SAGEConv


class GINNodeClassifier(nn.Module):
    """Three-layer GIN target model used for transfer-attack evaluation."""

    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                GINConv(self._make_mlp(nfeat, nhid)),
                GINConv(self._make_mlp(nhid, nhid)),
                GINConv(self._make_mlp(nhid, nhid)),
            ]
        )
        self.lin1 = nn.Linear(nhid * 3, nhid * 3)
        self.lin2 = nn.Linear(nhid * 3, nclass)
        self.dropout = dropout

    @staticmethod
    def _make_mlp(in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        hidden_states = []
        for conv in self.convs:
            x = conv(x, edge_index)
            hidden_states.append(x)
        x = torch.cat(hidden_states, dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.lin2(x), dim=1)


class GraphSAGENodeClassifier(nn.Module):
    """Three-layer GraphSAGE target model used for transfer evaluation."""

    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nhid)
        self.conv3 = SAGEConv(nhid, nclass)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


@dataclass(frozen=True)
class TargetEvaluation:
    loss: float
    accuracy: float


def build_target_model(model_name, nfeat, nhid, nclass, dropout=0.5):
    normalized_name = model_name.lower()
    if normalized_name == "gin":
        return GINNodeClassifier(nfeat, nhid, nclass, dropout)
    if normalized_name in {"gsage", "graphsage"}:
        return GraphSAGENodeClassifier(nfeat, nhid, nclass, dropout)
    raise ValueError(f"Unsupported target model: {model_name}")


def build_pyg_data(adj, features, labels, idx_train, idx_val):
    x = _features_to_tensor(features)
    edge_index = _adjacency_to_edge_index(adj)
    y = _labels_to_tensor(labels)
    num_nodes = x.shape[0]

    edge_out_of_range = bool(
        edge_index.numel() and edge_index.max().item() >= num_nodes
    )
    if y.shape[0] != num_nodes or edge_out_of_range:
        raise ValueError("Adjacency, features, and labels must use the same nodes")

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=_indices_to_mask(idx_train, num_nodes),
        val_mask=_indices_to_mask(idx_val, num_nodes),
    )


def train_and_evaluate_target(
    *,
    target_model_name,
    adj,
    features,
    labels,
    idx_train,
    idx_val,
    idx_eval,
    device,
    hidden=8,
    dropout=0.5,
    learning_rate=0.01,
    weight_decay=5e-4,
    epochs=200,
    patience=30,
    seed=15,
):
    _set_seed(seed)
    data = build_pyg_data(adj, features, labels, idx_train, idx_val).to(device)
    model = build_target_model(
        target_model_name,
        nfeat=data.x.shape[1],
        nhid=hidden,
        nclass=int(data.y.max().item()) + 1,
        dropout=dropout,
    ).to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    remaining_patience = max(1, int(patience))

    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            validation_mask = data.val_mask
            if not bool(validation_mask.any()):
                validation_mask = data.train_mask
            loss_val = F.nll_loss(
                output[validation_mask], data.y[validation_mask]
            ).item()

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = deepcopy(model.state_dict())
            remaining_patience = max(1, int(patience))
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    eval_indices = _indices_to_tensor(idx_eval, data.num_nodes).to(device)
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[eval_indices], data.y[eval_indices])
        accuracy = (
            output[eval_indices].argmax(dim=1) == data.y[eval_indices]
        ).float().mean()

    return TargetEvaluation(loss=loss.item(), accuracy=accuracy.item())


def _features_to_tensor(features):
    if sp.issparse(features):
        array = features.toarray()
        return torch.from_numpy(np.asarray(array)).float()
    if torch.is_tensor(features):
        tensor = features.detach().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        return tensor.float()
    return torch.as_tensor(np.asarray(features), dtype=torch.float32)


def _adjacency_to_edge_index(adj):
    if sp.issparse(adj):
        coo = adj.tocoo()
        nonzero = np.asarray(coo.data) != 0
        indices = np.vstack((coo.row[nonzero], coo.col[nonzero])).astype(np.int64)
        return torch.from_numpy(indices).long()
    if torch.is_tensor(adj):
        tensor = adj.detach().cpu()
        if tensor.layout != torch.strided:
            coalesced = tensor.to_sparse_coo().coalesce()
            return coalesced.indices()[:, coalesced.values() != 0].long()
        return tensor.nonzero(as_tuple=False).t().contiguous().long()

    array = np.asarray(adj)
    return torch.from_numpy(np.vstack(np.nonzero(array))).long()


def _labels_to_tensor(labels):
    if torch.is_tensor(labels):
        return labels.detach().cpu().reshape(-1).long()
    return torch.as_tensor(np.asarray(labels), dtype=torch.long).reshape(-1)


def _indices_to_tensor(indices, num_nodes):
    if torch.is_tensor(indices):
        tensor = indices.detach().cpu().reshape(-1).long()
    else:
        tensor = torch.as_tensor(np.asarray(indices), dtype=torch.long).reshape(-1)
    if tensor.numel() and (tensor.min() < 0 or tensor.max() >= num_nodes):
        raise IndexError("Node index is outside the graph")
    return tensor


def _indices_to_mask(indices, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[_indices_to_tensor(indices, num_nodes)] = True
    return mask


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
