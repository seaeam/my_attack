import torch
import networkx as nx
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.datasets import Coauthor
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon
from deeprobust.graph.data import Dataset


def get_planetoid_dataset(
    name, normalize_features=False, transform=None, split="normal", require_lcc=True
):
    # 顺序划分
    if split == "complete":
        dataset = Planetoid(root="./Data/", name=name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[: dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[
            dataset[0].num_nodes - 1000 : dataset[0].num_nodes - 500
        ] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500 :] = 1
    # 按比例划分
    elif split == "normal":
        data = Planetoid(root="./Data/", name=name)
        if require_lcc:  # 是否获取最大连通分支
            dataset = get_largest_connected_component(data)
        else:
            dataset = data[0]
        num_nodes = dataset.num_nodes
        # 6:2:2 划分
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)

        # 使用顺序索引划分训练集、验证集和测试集的索引
        idx_train = torch.arange(0, num_train)  # 选择前 60% 作为训练集
        idx_val = torch.arange(
            num_train, num_train + num_val
        )  # 选择接下来的 20% 作为验证集
        idx_test = torch.arange(num_train + num_val, num_nodes)  # 剩余 20% 作为测试集
        dataset.idx_train = idx_train
        dataset.idx_val = idx_val
        dataset.idx_test = idx_test

        # 更新掩码
        dataset.train_mask.fill_(False)
        dataset.val_mask.fill_(False)
        dataset.test_mask.fill_(False)

        # 使用划分的索引更新掩码
        dataset.train_mask[idx_train] = True
        dataset.val_mask[idx_val] = True
        dataset.test_mask[idx_test] = True

    # 如果需要进行特征归一化，并且有额外的数据转换操作
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def get_coauthor_dataset(
    name, normalize_features=False, transform=None, split="normal"
):
    if split == "complete":
        dataset = Coauthor(root="./Data/", name=name)
        new_dataset = Data(
            edge_index=dataset[0].edge_index, x=dataset[0].x, y=dataset[0].y
        )
        train_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        train_mask[: dataset[0].num_nodes - 1000] = 1
        val_mask[dataset[0].num_nodes - 1000 : dataset[0].num_nodes - 500] = 1
        test_mask[dataset[0].num_nodes - 500 :] = 1
        new_dataset.train_mask = train_mask
        new_dataset.val_mask = val_mask
        new_dataset.test_mask = test_mask
    elif split == "normal":
        dataset = Coauthor(root="./Data/", name=name)
        new_dataset = Data(
            edge_index=dataset[0].edge_index, x=dataset[0].x, y=dataset[0].y
        )
        train_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        num_nodes = dataset[0].num_nodes

        # 6:2:2 划分
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)

        train_mask[:num_train] = True
        val_mask[num_train : num_train + num_val] = True
        test_mask[num_train + num_val :] = True
        new_dataset.train_mask = train_mask
        new_dataset.val_mask = val_mask
        new_dataset.test_mask = test_mask

        # 使用顺序索引划分训练集、验证集和测试集的索引
        idx_train = torch.arange(0, num_train)  # 选择前 60% 作为训练集
        idx_val = torch.arange(
            num_train, num_train + num_val
        )  # 选择接下来的 20% 作为验证集
        idx_test = torch.arange(num_train + num_val, num_nodes)  # 剩余 20% 作为测试集
        new_dataset.idx_train = idx_train
        new_dataset.idx_val = idx_val
        new_dataset.idx_test = idx_test

        # 更新掩码
        new_dataset.train_mask.fill_(False)
        new_dataset.val_mask.fill_(False)
        new_dataset.test_mask.fill_(False)

        # 使用划分的索引更新掩码
        new_dataset.train_mask[idx_train] = True
        new_dataset.val_mask[idx_val] = True
        new_dataset.test_mask[idx_test] = True

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return new_dataset


def get_amazon_dataset(
    name, normalize_features=False, transform=None, split="normal", require_lcc=True
):
    # 顺序划分
    if split == "complete":
        dataset = Amazon(root="./Data/", name=name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[: dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[
            dataset[0].num_nodes - 1000 : dataset[0].num_nodes - 500
        ] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500 :] = 1
    # 按比例划分
    elif split == "normal":
        data = Amazon(root="./Data/", name=name)
        if require_lcc:  # 是否获取最大连通分支
            dataset = get_largest_connected_component(data)
        else:
            dataset = data[0]
        num_nodes = dataset.num_nodes
        # 6:2:2 划分
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)

        # 使用顺序索引划分训练集、验证集和测试集的索引
        idx_train = torch.arange(0, num_train)  # 选择前 60% 作为训练集
        idx_val = torch.arange(
            num_train, num_train + num_val
        )  # 选择接下来的 20% 作为验证集
        idx_test = torch.arange(num_train + num_val, num_nodes)  # 剩余 20% 作为测试集
        dataset.idx_train = idx_train
        dataset.idx_val = idx_val
        dataset.idx_test = idx_test

        # 初始化全 False 的掩码
        dataset.train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
        dataset.val_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
        dataset.test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)

        # 使用划分的索引更新掩码
        dataset.train_mask[idx_train] = True
        dataset.val_mask[idx_val] = True
        dataset.test_mask[idx_test] = True

    # 如果需要进行特征归一化，并且有额外的数据转换操作
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def get_deeproubust_dataset(
    name,
    normalize_features=False,
    transform=None,
    split="normal",
    require_lcc=True,
    seed=None,
):
    # 顺序划分
    if split == "complete":
        dataset = Dataset(root="./Data/", name=name, seed=seed)
        dataset.num_nodes = dataset.adj.shape[0]
        num_nodes = dataset.num_nodes

        idx_train = torch.arange(0, max(num_nodes - 1000, 0))
        idx_val = torch.arange(max(num_nodes - 1000, 0), max(num_nodes - 500, 0))
        idx_test = torch.arange(max(num_nodes - 500, 0), num_nodes)

        dataset.idx_train = idx_train
        dataset.idx_val = idx_val
        dataset.idx_test = idx_test

        dataset.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        dataset.train_mask[idx_train] = True
        dataset.val_mask[idx_val] = True
        dataset.test_mask[idx_test] = True
    # 按比例划分
    elif split == "normal":
        dataset = Dataset(root="./Data/", name=name, seed=seed)
        # DeepRobust's default Dataset(setting="nettack") already keeps the largest
        # connected component in this code path.
        # Do not pass it to the PyG-only LCC helper.
        dataset.num_nodes = dataset.adj.shape[0]
        num_nodes = dataset.num_nodes

        # Use a local generator so the requested split is reproducible without
        # consuming or depending on the process-wide torch RNG state.
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        idx_rand = torch.randperm(num_nodes, generator=generator)

        # 按6:2:2划分
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)

        # 使用随机索引划分
        idx_train = idx_rand[:num_train]  # 前60%训练集
        idx_val = idx_rand[num_train : num_train + num_val]  # 中间20%验证集
        idx_test = idx_rand[num_train + num_val :]  # 剩余20%测试集

        dataset.idx_train = idx_train
        dataset.idx_val = idx_val
        dataset.idx_test = idx_test

        # 初始化全False的掩码
        dataset.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # 更新掩码
        dataset.train_mask[idx_train] = True
        dataset.val_mask[idx_val] = True
        dataset.test_mask[idx_test] = True

    # 如果需要进行特征归一化，并且有额外的数据转换操作
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def get_largest_connected_component(dataset):
    # 获取 PyTorch Geometric 数据集中的边索引和节点数量
    edge_index = dataset[0].edge_index
    num_nodes = dataset[0].num_nodes
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # 将边的信息转换为邻接矩阵
    for i, j in edge_index.t().tolist():
        adj[i, j] = 1
        adj[j, i] = 1  # 因为是无向图，所以对称的边也设置为 1

    # 创建一个 NetworkX 图
    G = nx.from_numpy_array(adj.numpy())

    # 找到图中所有的连通分支
    connected_components = list(nx.connected_components(G))

    # 获取最大连通分支
    largest_component = max(connected_components, key=len)

    # 创建并返回最大连通分支的子图
    subgraph = G.subgraph(largest_component).copy()

    # 提取子图的节点索引
    subgraph_nodes = list(subgraph.nodes())

    # 获取子图中的边
    subgraph_edges = list(subgraph.edges())

    # 获取原始数据集中的节点特征
    x = dataset[0].x[subgraph_nodes]  # 选择最大连通分支中的节点特征
    y = dataset[0].y[subgraph_nodes]  # 选择最大连通分支中的节点标签
    edge_index_subgraph = (
        torch.tensor(
            [
                [subgraph_nodes.index(u), subgraph_nodes.index(v)]
                for u, v in subgraph_edges
            ],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )  # 将边转换为 PyTorch Geometric 格式的 edge_index

    # 获取掩码 (train_mask, val_mask, test_mask) 仅保留最大连通分支节点
    train_mask = dataset[0].train_mask[subgraph_nodes]
    val_mask = dataset[0].val_mask[subgraph_nodes]
    test_mask = dataset[0].test_mask[subgraph_nodes]

    # 创建一个新的 PyTorch Geometric 数据对象
    data = Data(
        x=x,
        edge_index=edge_index_subgraph,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data
