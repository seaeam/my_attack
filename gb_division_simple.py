"""
简化的粒球划分：直接在嵌入空间上进行粒球划分，类似 KMeans 的效果
用于替代 KMeans 进行层次化图攻击中的节点聚类
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist


def gb_division_on_embeddings(
    embeddings, target_clusters, mode="euclidean", max_iter=100, tol=1e-4
):
    """
    直接在嵌入向量上进行粒球划分，使用粒球的思想

    参数:
        embeddings: np.ndarray, shape (n, d) - 节点嵌入向量
        target_clusters: int - 目标粒球/簇数量
        mode: str - 距离度量方式 ('euclidean' 或 'cosine')
        max_iter: int - 最大迭代次数
        tol: float - 收敛阈值

    返回:
        cluster_ids: np.ndarray, shape (n,) - 每个点的簇ID
        centers: np.ndarray, shape (K, d) - 粒球中心
        radii: np.ndarray, shape (K,) - 粒球半径
    """
    n, d = embeddings.shape

    # 检查输入数据是否有效
    if n == 0 or d == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, d)), np.zeros(0)

    # 检查是否有 NaN 或 Inf
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        # 替换 NaN 和 Inf 为 0
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    K = min(target_clusters, n)  # 实际簇数不超过点数

    if K <= 0:
        return (
            np.zeros(n, dtype=np.int64),
            embeddings.mean(axis=0, keepdims=True),
            np.array([1.0]),
        )

    if K == 1:
        center = embeddings.mean(axis=0, keepdims=True)
        if mode == "cosine":
            # 余弦距离
            center_norm = center / (np.linalg.norm(center) + 1e-12)
            emb_norm = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            )
            dists = 1.0 - (emb_norm @ center_norm.T).squeeze()
        else:
            dists = np.linalg.norm(embeddings - center, axis=1)
        radius = dists.max()
        return np.zeros(n, dtype=np.int64), center, np.array([radius])

    # 初始化：K-means++ 风格
    centers = _kmeans_plusplus_init(embeddings, K, mode)

    cluster_ids = np.zeros(n, dtype=np.int64)

    for iteration in range(max_iter):
        old_centers = centers.copy()

        # E-step: 分配点到最近的中心
        if mode == "cosine":
            # 余弦相似度
            centers_norm = centers / (
                np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
            )
            emb_norm = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            )
            similarities = emb_norm @ centers_norm.T
            cluster_ids = similarities.argmax(axis=1)
        else:
            # 欧氏距离
            dists = cdist(embeddings, centers, metric="euclidean")
            cluster_ids = dists.argmin(axis=1)

        # M-step: 更新中心（加权平均，考虑密度）
        empty_clusters = []
        for k in range(K):
            mask = cluster_ids == k
            if mask.sum() > 0:
                # 粒球中心：该簇所有点的均值
                centers[k] = embeddings[mask].mean(axis=0)
            else:
                empty_clusters.append(k)

        # 如果有空簇，重新初始化（选择距离现有中心最远的点）
        if empty_clusters:
            for k in empty_clusters:
                if mode == "cosine":
                    centers_norm = centers / (
                        np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
                    )
                    emb_norm = embeddings / (
                        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
                    )
                    sims = emb_norm @ centers_norm.T
                    max_sims = sims.max(axis=1)
                    # 避免选择已经很近的点
                    if max_sims.min() < 0.99:  # 不是所有点都已经很接近
                        farthest = max_sims.argmin()
                    else:
                        farthest = np.random.randint(n)
                else:
                    dists_to_centers = cdist(embeddings, centers, metric="euclidean")
                    min_dists = dists_to_centers.min(axis=1)
                    # 避免数值问题
                    if min_dists.max() > 1e-6:
                        farthest = min_dists.argmax()
                    else:
                        farthest = np.random.randint(n)
                centers[k] = embeddings[farthest]

        # 检查收敛
        center_shift = np.linalg.norm(centers - old_centers)
        if center_shift < tol:
            break

    # 计算每个粒球的半径（到中心的最大距离）
    radii = np.zeros(K)
    for k in range(K):
        mask = cluster_ids == k
        if mask.sum() > 0:
            if mode == "cosine":
                center_norm = centers[k : k + 1] / (
                    np.linalg.norm(centers[k : k + 1]) + 1e-12
                )
                emb_norm = embeddings[mask] / (
                    np.linalg.norm(embeddings[mask], axis=1, keepdims=True) + 1e-12
                )
                dists = 1.0 - (emb_norm @ center_norm.T).squeeze()
            else:
                dists = np.linalg.norm(embeddings[mask] - centers[k], axis=1)
            radii[k] = dists.max() if dists.size > 0 else 0.0
        else:
            radii[k] = 0.0

    return cluster_ids, centers, radii


def _kmeans_plusplus_init(embeddings, K, mode="euclidean"):
    """K-means++ 初始化策略"""
    n, d = embeddings.shape
    centers = np.zeros((K, d))

    # 随机选择第一个中心
    centers[0] = embeddings[np.random.randint(n)]

    for k in range(1, K):
        # 计算每个点到已有中心的最小距离
        if mode == "cosine":
            centers_norm = centers[:k] / (
                np.linalg.norm(centers[:k], axis=1, keepdims=True) + 1e-12
            )
            emb_norm = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            )
            sims = emb_norm @ centers_norm.T
            min_dists = 1.0 - sims.max(axis=1)
        else:
            dists = cdist(embeddings, centers[:k], metric="euclidean")
            min_dists = dists.min(axis=1)

        # 以距离平方作为概率选择下一个中心
        probs = min_dists**2
        probs = np.maximum(probs, 1e-12)  # 避免全零
        prob_sum = probs.sum()

        # 如果所有概率都是0或NaN，使用均匀分布
        if prob_sum <= 1e-12 or np.isnan(prob_sum) or np.isinf(prob_sum):
            next_center_idx = np.random.randint(n)
        else:
            probs /= prob_sum
            # 再次检查是否有 NaN
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                next_center_idx = np.random.randint(n)
            else:
                next_center_idx = np.random.choice(n, p=probs)

        centers[k] = embeddings[next_center_idx]

    return centers


def gb_division_simple(embeddings, target_clusters, mode="euclidean"):
    """
    简化接口：只返回簇ID

    参数:
        embeddings: torch.Tensor or np.ndarray, shape (n, d)
        target_clusters: int - 目标簇数
        mode: str - 'euclidean' 或 'cosine'

    返回:
        cluster_ids: torch.Tensor, shape (n,) - 每个点的簇ID
    """
    # 转换为 numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    # 调用粒球划分
    cluster_ids, centers, radii = gb_division_on_embeddings(
        embeddings_np, target_clusters, mode=mode
    )

    # 转换回 torch
    return torch.from_numpy(cluster_ids).long()


# 兼容接口：模拟 KMeans 的 fit_predict 方法
class GBCluster:
    """粒球聚类器，提供类似 KMeans 的接口"""

    def __init__(self, n_clusters, mode="euclidean", verbose=0):
        self.n_clusters = n_clusters
        self.mode = mode
        self.verbose = verbose
        self.centroids = None
        self.radii = None

    def fit_predict(self, embeddings, centroids=None):
        """
        对嵌入向量进行粒球聚类

        参数:
            embeddings: torch.Tensor, shape (n, d)
            centroids: torch.Tensor, optional - 预先计算的中心（用于warm-start）

        返回:
            cluster_ids: torch.Tensor, shape (n,) - 簇ID
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = embeddings

        # 如果提供了中心点，使用它们进行分配
        if centroids is not None:
            if isinstance(centroids, torch.Tensor):
                centers_np = centroids.detach().cpu().numpy()
            else:
                centers_np = centroids

            # 直接分配到最近的中心
            if self.mode == "cosine":
                centers_norm = centers_np / (
                    np.linalg.norm(centers_np, axis=1, keepdims=True) + 1e-12
                )
                emb_norm = embeddings_np / (
                    np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-12
                )
                similarities = emb_norm @ centers_norm.T
                cluster_ids = similarities.argmax(axis=1)
            else:
                dists = cdist(embeddings_np, centers_np, metric="euclidean")
                cluster_ids = dists.argmin(axis=1)

            self.centroids = torch.from_numpy(centers_np).float()
            return torch.from_numpy(cluster_ids).long()

        # 否则重新聚类
        cluster_ids, centers, radii = gb_division_on_embeddings(
            embeddings_np, self.n_clusters, mode=self.mode
        )

        self.centroids = torch.from_numpy(centers).float()
        self.radii = torch.from_numpy(radii).float()

        return torch.from_numpy(cluster_ids).long()
