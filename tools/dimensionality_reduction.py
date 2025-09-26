import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv

def SVD_reduction(gb_data, adj):
    """
    使用奇异值分解(SVD)对邻接矩阵进行降维，并将降维后的特征矩阵赋值给 gb_data.x。

    参数:
    - gb_data: 图数据对象，包含节点特征矩阵 x 和其他图相关信息。
    - adj: 邻接矩阵，形状为 (2708, 2708)，即图的结构信息。
    - target_dim: 目标降维后的特征维度，默认为1433。
    """
    # 更新 target_dim 为 gb_data.x 的特征维度数
    target_dim = gb_data.x.size(1)
    # 确保 adj 是密集矩阵
    adj_dense = adj.to_dense()  # 将稀疏矩阵转换为密集矩阵

    # 使用奇异值分解（SVD）对 adj 进行降维
    U, S, V = torch.linalg.svd(adj_dense, full_matrices=False)

    # 选择前 target_dim 个奇异值和对应的 U, S, V 矩阵
    U_reduced = U[:, :target_dim]
    S_reduced = S[:target_dim]
    V_reduced = V[:, :target_dim]

    # 通过 U_reduced 和 S_reduced 生成降维后的特征矩阵
    x_reduced = torch.matmul(U_reduced, torch.diag(S_reduced))

    # 更新 gb_data 的 x 特征矩阵为降维后的结果
    gb_data.x = x_reduced

    # 更新 gb_data 的 edge_index
    edge_index = adj_dense.nonzero(as_tuple=True)
    gb_data.edge_index = torch.stack(edge_index, dim=0)

    return gb_data


def PCA_reduction(gb_data, adj):
    """
    使用PCA对邻接矩阵进行降维，并将降维后的特征矩阵赋值给 gb_data.x。

    参数:
    - gb_data: 图数据对象，包含节点特征矩阵 x 和其他图相关信息。
    - adj: 邻接矩阵，形状为 (2708, 2708)，即图的结构信息。
    - target_dim: 目标降维后的特征维度，默认为1433。
    """
    # 更新 target_dim 为 gb_data.x 的特征维度数
    target_dim = gb_data.x.size(1)
    # 确保 adj 是密集矩阵
    adj_dense = adj.to_dense()  # 将稀疏矩阵转换为密集矩阵

    # 将邻接矩阵转换为 NumPy 数组，因为 PCA 需要 NumPy 格式
    adj_numpy = adj_dense.numpy()

    # 使用 PCA 进行降维
    pca = PCA(n_components=target_dim)

    # 对邻接矩阵进行降维，转换后的矩阵形状为 (2708, target_dim)
    adj_reduced = pca.fit_transform(adj_numpy)

    # 将降维后的结果转换回 torch.Tensor 并赋值给 gb_data.x
    gb_data.x = torch.tensor(adj_reduced, dtype=torch.float32)

    # 更新 gb_data 的 edge_index，保留原有的邻接关系
    edge_index = adj_dense.nonzero(as_tuple=True)
    gb_data.edge_index = torch.stack(edge_index, dim=0)

    return gb_data


def GCN_reduction(gb_data, adj):
    """
    使用 GCN 对邻接矩阵进行降维，并将降维后的特征矩阵赋值给 gb_data.x。

    参数:
    - gb_data: 图数据对象，包含节点特征矩阵 x 和其他图相关信息。
    - adj: 邻接矩阵，即图的结构信息。

    返回:
    - gb_data: 更新后的图数据对象，包含降维后的特征矩阵 x 和边索引 edge_index。
    """
    # 获取目标维度
    target_dim = gb_data.x.size(1)

    # 确保 adj 是稀疏矩阵
    if adj.is_sparse:
        edge_index = adj.coalesce().indices()  # 直接从稀疏矩阵获取边的索引
    else:
        # 如果 adj 是密集矩阵，首先转换为稀疏格式
        adj_sparse = adj.to_sparse()
        edge_index = adj_sparse.coalesce().indices()  # 获取稀疏矩阵的边索引

    features = adj  # 使用图数据中的原始特征作为输入特征

    # 创建一个 GCN 模型，输出特征维度为 target_dim
    class GCN_DimReduce(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GCN_DimReduce, self).__init__()
            self.conv1 = GCNConv(in_channels, 64)  # 第一层，输出为 64 维
            self.conv2 = GCNConv(64, out_channels)  # 第二层，输出为 target_dim

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)  # ReLU 激活
            x = self.conv2(x, edge_index)
            return x

    # 创建 GCN 模型
    model = GCN_DimReduce(in_channels=features.size(1), out_channels=target_dim)
    model = model.to(features.device)  # 将模型移到相同设备

    # 在评估模式下获取降维后的特征
    model.eval()
    with torch.no_grad():
        reduced_features = model(features, edge_index)  # 获取降维后的节点特征

    # 将降维后的特征赋值给 gb_data.x，转换为 NumPy 数组
    gb_data.x = reduced_features.cpu()  # 转换为 NumPy 数组

    # 更新 gb_data 的 edge_index，保留原有的邻接关系
    adj_dense = adj.to_dense()
    edge_index = adj_dense.nonzero(as_tuple=True)
    gb_data.edge_index = torch.stack(edge_index, dim=0)

    return gb_data


import torch

def top_k_reduction(gb_data, adj):
    """
    使用 Top-k 策略降维邻接矩阵，将一个 n x n 的邻接矩阵降维为 n x k 的邻接矩阵。

    参数：
    adj (torch.Tensor): 输入的邻接矩阵 (n x n)，可以是稀疏矩阵
    gb_data (object): 图数据对象，包含节点特征，用于获取目标维度 k
    k (int): 保留的邻居数量

    返回：
    torch.Tensor: 降维后的邻接矩阵 (n x k)
    """
    # 确保 adj 是稀疏矩阵，并将其转换为稠密矩阵
    if adj.is_sparse:
        adj_dense = adj.to_dense()  # 将稀疏矩阵转换为稠密矩阵
    else:
        adj_dense = adj  # 如果已经是稠密矩阵，直接使用

    # 更新 gb_data 的 edge_index，保留原有的邻接关系
    adj_dense = adj.to_dense()
    edge_index = adj_dense.nonzero(as_tuple=True)
    gb_data.edge_index = torch.stack(edge_index, dim=0)

    n = adj_dense.size(0)  # 获取节点数量 (n x n)
    k = gb_data.x.size(1)  # 获取目标维度 k
    top_k_adj = torch.zeros(n, k).to(adj.device)  # 初始化结果矩阵 (n x k)

    # 对每一行进行 top-k 操作
    for i in range(n):
        row = adj_dense[i]  # 获取第 i 行的邻接信息
        _, top_k_indices = torch.topk(row, k, largest=True, sorted=False)  # 获取前 k 个邻居的索引
        top_k_adj[i] = row[top_k_indices]  # 赋值到 top_k_adj 矩阵

        # 将降维后的特征赋值给 gb_data.x，转换为 NumPy 数组
    gb_data.x = top_k_adj.cpu()  # 转换为 NumPy 数组

    return gb_data
