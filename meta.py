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
parser.add_argument("--miter", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=16, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
)
parser.add_argument("--dataset", type=str, default="citeseer", help="dataset")
parser.add_argument("--ptb_rate", type=float, default=0.05, help="pertubation rate")
parser.add_argument(
    "--model",
    type=str,
    default="Meta-Both",
    choices=["Meta-Both", "Meta-Self", "Meta-Train"],
    help="model variant",
)
parser.add_argument(
    "--use_moe", action="store_true", default=True, help="Enable MoE enhancement"
)
parser.add_argument(
    "--no_moe", action="store_true", default=False, help="Disable MoE enhancement"
)
parser.add_argument(
    "--moe_weight", type=float, default=1, help="MoE gradient weight [0, 1]"
)

args = parser.parse_args()

device = torch.device(
    "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != "cpu":
    torch.cuda.manual_seed(args.seed)


pyg_flag = False


amazon = "Computers"
# amazon = 'Photo'

if args.dataset == "physics" or args.dataset == "cs":
    gb_data = get_coauthor_dataset(args.dataset, split=args.split_data)
elif args.dataset == "acm" or args.dataset == "cora_ml" or args.dataset == "polblogs":
    gb_data = get_deeproubust_dataset(args.dataset, split=args.split_data)
elif args.dataset == "computers" or args.dataset == "photo":
    gb_data = get_amazon_dataset(args.dataset, split=args.split_data)
else:
    gb_data = get_planetoid_dataset(args.dataset, split=args.split_data)

if hasattr(
    gb_data, "adj"
):  # get_deeproubust_dataset中的数据集需要额外添加x、y、edge_index
    gb_data.x = torch.from_numpy(gb_data.features.toarray()).float()
    gb_data.y = torch.from_numpy(gb_data.labels).long()
    adj_coo = gb_data.adj.tocoo()
    edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
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
    # assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"
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

lambda_ = 1

model = Heirattack(
    model=surrogate,
    nnodes=adj.shape[0],
    feature_shape=features.shape,
    attack_structure=True,
    attack_features=True,
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

# 设置 MoE 参数
model.use_moe_default = args.use_moe and not args.no_moe
model.moe_grad_weight_default = args.moe_weight
print(f"\n🤖 MoE Status: {'✅ ENABLED' if model.use_moe_default else '❌ DISABLED'}")
if model.use_moe_default:
    print(f"   MoE gradient weight: {model.moe_grad_weight_default:.2f}")
#
# model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
#                    attack_features=False, device=device, lambda_=0.5)

# model = DICE()

model = model.to(device)


def test(adj, features, idx_eval, description="Test set"):
    """Test GCN and ensure tensors are on CPU for deeprobust."""
    import scipy.sparse as sp

    # 将 features 和 adj 转到 CPU
    if torch.is_tensor(features) and features.is_cuda:
        features = features.cpu()
    if torch.is_tensor(adj) and adj.is_cuda:
        adj = adj.cpu()
    elif sp.issparse(adj):
        adj = adj.tocoo()  # 保留 sparse 格式

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
    # 执行多步攻击
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

    # 取攻击结果
    modified_adj = model.modified_adj
    modified_features = model.modified_features

    print("=== Clean graph ===")
    test(adj, features, idx_unlabeled, description="Clean graph")

    print("=== Edge-only attack ===")
    test(modified_adj, features, idx_unlabeled, description="Edge-only attack")

    print("=== Feature-only attack ===")
    test(adj, modified_features, idx_unlabeled, description="Feature-only attack")

    print("=== Combined attack (edge + feature) ===")
    test(modified_adj, modified_features, idx_unlabeled, description="Combined attack")

    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root="./", name=f"mod_adj_polblogs_005_metatrain")
    # model.save_features(root="./", name="mod_features")


if __name__ == "__main__":
    main()
