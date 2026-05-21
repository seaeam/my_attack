"""
消融实验：仅边攻击（Structure-only attack）
不启用任何特征/文本攻击，attack_features=False
"""

import copy
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from heir import Heirattack
from deeprobust.graph.global_attack import DICE, MetaApprox

import math
import sys
import os
import datetime


class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


from split_test import (
    get_amazon_dataset,
    get_coauthor_dataset,
    get_deeproubust_dataset,
    get_planetoid_dataset,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument("--split_data", type=str, default="normal")
parser.add_argument("--oracle", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=15, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of epochs to train."
)
parser.add_argument("--ball_r", type=float, default=0.8)
parser.add_argument("--noise", type=int, default=0)
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--level", type=int, default=2)
parser.add_argument(
    "--coarsen_method",
    type=str,
    default="gb",
    choices=["gb", "kmeans"],
)
parser.add_argument("--miter", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--global_important_ratio",
    type=float,
    default=0.10,
)
parser.add_argument(
    "--global_ppr_alpha",
    type=float,
    default=0.15,
)
parser.add_argument(
    "--global_ppr_iters",
    type=int,
    default=30,
)
parser.add_argument(
    "--global_seed_strategy",
    type=str,
    default="uniform",
    choices=["uniform", "degree", "label"],
)
parser.add_argument("--global_label_weight", type=float, default=0.0)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
)
parser.add_argument("--hidden", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--dataset", type=str, default="citeseer")
parser.add_argument("--ptb_rate", type=float, default=0.05)
parser.add_argument(
    "--model",
    type=str,
    default="Meta-Both",
    choices=["Meta-Both", "Meta-Self", "Meta-Train"],
)

args = parser.parse_args()

# 消融实验强制关闭文本攻击相关参数
args.use_text_attack = False
args.freeze_structure_features = False
args.text_attack_nodes = None
args.text_attack_max_visits = 1
args.text_retries = 0
args.text_budget_per_node = 15
args.text_topk_ratio = 0.05
args.text_ppr_alpha = 0.20
args.text_ppr_iters = 25
args.text_min_cluster_size = 2
args.text_max_cluster_size = 8
args.text_similarity_min = 0.85
args.text_cdl_topk = 10
args.text_cluster_attr_topk = 10
args.text_max_added_words = 20

# Setup logger
if not os.path.exists("logs"):
    os.makedirs("logs")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/ablation_edge_{args.dataset}_{args.model}_{timestamp}.txt"
sys.stdout = Logger(log_filename)

print("=" * 60)
print("  Ablation: Edge-Only Attack (no feature/text attack)")
print("=" * 60)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != "cpu":
    torch.cuda.manual_seed(args.seed)

pyg_flag = False
amazon = "Computers"

deeprobust_datasets = {"acm", "cora_ml", "polblogs", "blogcatalog", "uai", "flickr"}

if args.dataset == "physics" or args.dataset == "cs":
    gb_data = get_coauthor_dataset(args.dataset, split=args.split_data)
elif args.dataset in deeprobust_datasets:
    gb_data = get_deeproubust_dataset(args.dataset, split=args.split_data)
elif args.dataset == "computers" or args.dataset == "photo":
    gb_data = get_amazon_dataset(args.dataset, split=args.split_data)
else:
    gb_data = get_planetoid_dataset(args.dataset, split=args.split_data)

if hasattr(gb_data, "adj"):
    gb_data.x = torch.from_numpy(gb_data.features.toarray()).float()
    gb_data.y = torch.from_numpy(gb_data.labels).long()
    adj_coo = gb_data.adj.tocoo()
    edge_index = torch.from_numpy(
        np.vstack((adj_coo.row, adj_coo.col)).astype(np.int64)
    )
    gb_data.edge_index = edge_index

if "amazon" in args.dataset:
    from torch_geometric.datasets import Amazon

    dataset = Amazon(root="./Data/pygdata/", name=amazon)
    dataset = Amazon(root="./Data/", name="Photo")
    pyg_flag = True
elif "cs" in args.dataset:
    from torch_geometric.datasets import Coauthor

    dataset = Coauthor(root="./Data/", name=args.dataset)
    pyg_flag = True
elif "dblp" in args.dataset:
    from torch_geometric.datasets import DBLP, WikiCS

    dataset = DBLP(root="./Data/")
    pyg_flag = True
elif "reddit" in args.dataset:
    from torch_geometric.datasets import Reddit

    dataset = Reddit(root="/tmp")
    pyg_flag = True
elif "ogbn" in args.dataset:
    from ogb.nodeproppred import PygNodePropPredDataset
    from deeprobust.graph.data import Pyg2Dpr

    dataset = PygNodePropPredDataset(name=args.dataset)
    pyg_flag = True
else:
    data = Dataset(root="./Data/", name=args.dataset, setting="nettack")
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

if pyg_flag:
    if args.dataset == "amazon":
        from deeprobust.graph.data import AmazonPyg

        data = AmazonPyg("./Data/", name=amazon)
        from deeprobust.graph.data import Pyg2Dpr

        data = Pyg2Dpr(data)
    elif "cs" in args.dataset:
        from deeprobust.graph.data import CoauthorPyg

        data = CoauthorPyg("./Data/", name=args.dataset)
        from deeprobust.graph.data import Pyg2Dpr

        data = Pyg2Dpr(data)
    else:
        from deeprobust.graph.data import Pyg2Dpr

        data = Pyg2Dpr(dataset)

    adj, features, labels = data.adj, data.features, data.labels
    print(adj.sum(), labels.min(), labels.max())
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1
    lcc = data.largest_connected_components(adj)
    print(len(lcc))
    data.setting, data.seed = "nettack", args.seed
    idx_train, idx_val, idx_test = data.get_train_val_test()

org_adj = data.adj.copy()
print(
    features.shape,
    np.unique(features).shape,
    adj.shape,
    len(idx_train),
    len(idx_val),
    len(idx_test),
)

idx_unlabeled = np.union1d(idx_val, idx_test)
if "Self" in args.model:
    idx_attack = idx_unlabeled
if "Train" in args.model:
    idx_attack = idx_train
if "Both" in args.model:
    idx_attack = np.union1d(idx_train, idx_unlabeled)
lambda_ = 1

perturbations = int(args.ptb_rate * (adj.sum() // 2))
if pyg_flag:
    from scipy import sparse

    features = sparse.csr_matrix(features)
    adj, features, labels = preprocess(
        adj, features, labels, preprocess_adj=False, sparse=True
    )
    features = features.to_dense()
else:
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Surrogate Model
surrogate = GCN(
    nfeat=features.shape[1],
    nclass=labels.max().item() + 1,
    nhid=16,
    dropout=0.5,
    with_relu=False,
    with_bias=True,
    weight_decay=5e-4,
    device=device,
)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

output = surrogate.output.cpu()
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])
print(
    "Test set results:",
    "loss= {:.4f}".format(loss_test.item()),
    "accuracy= {:.4f}".format(acc_test.item()),
)

# ★ 关键：attack_features=False，纯结构攻击
model = Heirattack(
    model=surrogate,
    nnodes=adj.shape[0],
    feature_shape=features.shape,
    attack_structure=True,
    attack_features=False,
    device=device,
    lambda_=lambda_,
    train_iters=args.miter,
    levels=args.level,
    gb_data=gb_data,
    use_oracle=args.oracle,
    lr=args.lr,
    args=args,
    features=features,
)
model = model.to(device)


def test(adj, features, idx_eval, description="Test set"):
    import scipy.sparse as sp

    if torch.is_tensor(features) and features.is_cuda:
        features = features.cpu()
    if torch.is_tensor(adj) and adj.is_cuda:
        adj = adj.cpu()
    elif sp.issparse(adj):
        adj = adj.tocoo()

    gcn = GCN(
        nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=labels.max().item() + 1,
        dropout=args.dropout,
        device=device,
    )
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train)
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_eval], labels[idx_eval])
    acc_test = accuracy(output[idx_eval], labels[idx_eval])
    print(
        f"{description} results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
    )
    return acc_test.item()


def main():
    print(f"\n  Dataset: {args.dataset}")
    print(f"  Perturbation budget: {perturbations} edges ({args.ptb_rate*100}%)")
    print(f"  Attack mode: {args.model}")
    print(
        f"  Structure search: level={args.level}, step={args.step}, miter={args.miter}, lr={args.lr}"
    )
    print(
        f"  Global PPR: ratio={args.global_important_ratio}, alpha={args.global_ppr_alpha}, "
        f"iters={args.global_ppr_iters}, seed={args.global_seed_strategy}"
    )
    print(f"  Feature attack: DISABLED (edge-only ablation)")
    print()

    model.meta_attack_multi_step(
        features,
        org_adj,
        labels,
        idx_train,
        idx_unlabeled,
        perturbations,
        n_step=args.step,
        type=args.model,
    )

    modified_adj = model.modified_adj

    print("\n" + "=" * 60)
    print("  Ablation Results: Edge-Only Attack")
    print("=" * 60 + "\n")

    acc_clean = test(adj, features, idx_unlabeled, description="Clean graph")
    acc_edge = test(
        modified_adj, features, idx_unlabeled, description="Edge-only attack"
    )

    print("\n" + "=" * 60)
    print(f"  Clean accuracy:       {acc_clean:.4f}")
    print(f"  Edge attack accuracy: {acc_edge:.4f} (drop: {acc_clean - acc_edge:.4f})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
