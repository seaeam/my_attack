import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial.distance import cdist

from tools import del_outlier, get_dataset
from tools.add_id import add_id
from tools.add_purity import add_purity
from tools.corse_split import initial_splite
from tools.purification import purification
from tools.split_ball_purity import split_ball_further, split_ball_purity


def _compute_distances(points, centers, mode="euclidean"):
    if len(points) == 0 or len(centers) == 0:
        return np.zeros((len(points), len(centers)), dtype=np.float32)

    if mode == "cosine":
        points_norm = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-12)
        centers_norm = centers / (
            np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
        )
        return 1.0 - (points_norm @ centers_norm.T)

    return cdist(points, centers, metric="euclidean")


def _compute_radius(points, center, mode="euclidean"):
    if len(points) == 0:
        return 0.0

    dists = _compute_distances(points, center[None, :], mode=mode).reshape(-1)
    return float(dists.max()) if dists.size > 0 else 0.0


def _build_knn_adjacency(embeddings, mode="euclidean", num_neighbors=None):
    n = embeddings.shape[0]
    if n <= 1:
        return sp.coo_matrix((n, n), dtype=np.float32)

    if num_neighbors is None:
        num_neighbors = max(1, min(n - 1, int(np.sqrt(n))))

    dists = _compute_distances(embeddings, embeddings, mode=mode)
    np.fill_diagonal(dists, np.inf)

    rows, cols = [], []
    for node in range(n):
        neighbors = np.argsort(dists[node])[:num_neighbors]
        rows.extend([node] * len(neighbors))
        cols.extend(neighbors.tolist())

    data = np.ones(len(rows), dtype=np.float32)
    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    adjacency = adjacency.maximum(adjacency.T).tocoo()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def _kmeans_plusplus_init(embeddings, K, mode="euclidean"):
    n, d = embeddings.shape
    centers = np.zeros((K, d), dtype=np.float32)
    centers[0] = embeddings[np.random.randint(n)]

    for k in range(1, K):
        min_dists = _compute_distances(embeddings, centers[:k], mode=mode).min(axis=1)
        probs = np.maximum(min_dists**2, 1e-12)
        prob_sum = probs.sum()

        if prob_sum <= 1e-12 or np.isnan(prob_sum) or np.isinf(prob_sum):
            next_center_idx = np.random.randint(n)
        else:
            probs = probs / prob_sum
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                next_center_idx = np.random.randint(n)
            else:
                next_center_idx = np.random.choice(n, p=probs)

        centers[k] = embeddings[next_center_idx]

    return centers


def _bootstrap_pseudo_labels(
    embeddings, target_clusters, mode="euclidean", centroids=None
):
    n = embeddings.shape[0]
    K = min(max(int(target_clusters), 0), n)

    if n == 0 or K <= 1:
        return np.zeros(n, dtype=np.int64)

    if centroids is not None:
        centers = np.asarray(centroids, dtype=np.float32)
        if centers.ndim == 1:
            centers = centers.reshape(1, -1)
        if centers.shape[1] != embeddings.shape[1] or centers.shape[0] == 0:
            centers = None
    else:
        centers = None

    if centers is None:
        centers = _kmeans_plusplus_init(embeddings, K, mode=mode)

    dists = _compute_distances(embeddings, centers, mode=mode)
    return dists.argmin(axis=1).astype(np.int64)


def _cluster_stats_from_assignments(embeddings, cluster_ids, mode="euclidean"):
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    valid_ids = sorted(int(cid) for cid in np.unique(cluster_ids) if cid >= 0)

    if not valid_ids:
        cluster_ids = np.zeros(len(embeddings), dtype=np.int64)
        valid_ids = [0]

    remapped = cluster_ids.copy()
    centers = np.zeros((len(valid_ids), embeddings.shape[1]), dtype=np.float32)
    radii = np.zeros(len(valid_ids), dtype=np.float32)

    for new_id, old_id in enumerate(valid_ids):
        nodes = np.where(cluster_ids == old_id)[0]
        remapped[nodes] = new_id
        points = embeddings[nodes]
        centers[new_id] = points.mean(axis=0)
        radii[new_id] = _compute_radius(points, centers[new_id], mode=mode)

    return remapped, centers, radii


def gb_division_on_embeddings(
    embeddings,
    target_clusters,
    mode="euclidean",
    max_iter=100,
    tol=1e-4,
    centroids=None,
):
    """
    在嵌入向量上运行 `gb_division.py` 的粒球拆分流程。

    由于这里没有原始图和真实标签，内部会：
    1. 从 embedding 构造一个对称 kNN 图；
    2. 用 warm-start 中心或 k-means++ 初始化生成伪标签；
    3. 复用原版的初分、纯度细分、进一步细分和净化流程；
    4. 再整理回簇 ID、中心点和半径。
    """
    del max_iter, tol

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    n, d = embeddings.shape
    if n == 0 or d == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, d), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    K = min(max(int(target_clusters), 0), n)
    if K <= 0:
        return (
            np.zeros(n, dtype=np.int64),
            embeddings.mean(axis=0, keepdims=True),
            np.array([1.0], dtype=np.float32),
        )

    if K == 1:
        center = embeddings.mean(axis=0, keepdims=True)
        radius = _compute_radius(embeddings, center[0], mode=mode)
        return np.zeros(n, dtype=np.int64), center, np.array([radius], dtype=np.float32)

    adjacency = _build_knn_adjacency(embeddings, mode=mode)
    pseudo_labels = _bootstrap_pseudo_labels(
        embeddings, K, mode=mode, centroids=centroids
    )

    graph = get_dataset.get_dataset(embeddings, adjacency, pseudo_labels, seed=0)
    graph = del_outlier.del_outlier(graph)

    if graph.number_of_nodes() == 0:
        center = embeddings.mean(axis=0, keepdims=True)
        radius = _compute_radius(embeddings, center[0], mode=mode)
        return np.zeros(n, dtype=np.int64), center, np.array([radius], dtype=np.float32)

    total_degree_dict_old = dict(graph.degree())
    old_node_ids = list(total_degree_dict_old.keys())
    total_degree_dict = {
        new_id: deg for new_id, (_, deg) in enumerate(total_degree_dict_old.items())
    }
    id_dict = {new_id: old_id for new_id, old_id in enumerate(total_degree_dict_old)}
    id_dict_oldtonew = {
        old_id: new_id for new_id, old_id in enumerate(total_degree_dict_old)
    }

    attributes = np.asarray(
        [graph.nodes[old_id]["attributes"] for old_id in old_node_ids], dtype=np.float32
    )
    labels = np.asarray(
        [graph.nodes[old_id]["label"] for old_id in old_node_ids], dtype=np.float32
    ).reshape(-1, 1)
    indices = np.asarray(old_node_ids, dtype=np.int64).reshape(-1, 1)
    data_mat = np.concatenate((indices, attributes, labels), axis=1)
    data_mat = add_id(data_mat)

    clusters = [[data_mat, total_degree_dict]]
    granular_balls = initial_splite(
        clusters, graph, id_dict, id_dict_oldtonew, labels, total_degree_dict
    )

    if not granular_balls:
        return _cluster_stats_from_assignments(embeddings, pseudo_labels, mode=mode)

    dropped_nodes = []
    while len(granular_balls) > K:
        granular_balls.sort(key=lambda x: len(x[0]))
        cut_pos = len(granular_balls) - K

        for dropped_ball in granular_balls[:cut_pos]:
            rows = np.asarray(dropped_ball[0])
            if rows.size != 0:
                dropped_nodes.extend(rows[:, 0].astype(int).tolist())

        granular_balls = granular_balls[cut_pos:]

    granular_balls = add_purity(granular_balls)
    granular_balls = split_ball_purity(
        graph, id_dict, granular_balls, total_degree_dict, K
    )
    if len(granular_balls) < K:
        granular_balls = split_ball_further(
            graph, id_dict, granular_balls, total_degree_dict, K
        )
    granular_balls = purification(granular_balls)

    if not granular_balls:
        return _cluster_stats_from_assignments(embeddings, pseudo_labels, mode=mode)

    cluster_ids = np.full(n, -1, dtype=np.int64)
    provisional_centers = []
    for gb_idx, gb in enumerate(granular_balls):
        rows = np.asarray(gb[0])
        if rows.size == 0:
            continue
        nodes = rows[:, 0].astype(int)
        cluster_ids[nodes] = gb_idx
        provisional_centers.append(embeddings[nodes].mean(axis=0))

    if provisional_centers:
        provisional_centers = np.asarray(provisional_centers, dtype=np.float32)
    else:
        provisional_centers = np.zeros((0, d), dtype=np.float32)

    unassigned = np.where(cluster_ids < 0)[0]
    if unassigned.size > 0:
        if provisional_centers.shape[0] > 0:
            nearest = _compute_distances(
                embeddings[unassigned], provisional_centers, mode=mode
            ).argmin(axis=1)
            cluster_ids[unassigned] = nearest.astype(np.int64)
        else:
            cluster_ids[unassigned] = pseudo_labels[unassigned]

    if dropped_nodes:
        dropped_nodes = np.asarray(sorted(set(dropped_nodes)), dtype=np.int64)
        dropped_nodes = dropped_nodes[(dropped_nodes >= 0) & (dropped_nodes < n)]
        if dropped_nodes.size > 0 and provisional_centers.shape[0] > 0:
            nearest = _compute_distances(
                embeddings[dropped_nodes], provisional_centers, mode=mode
            ).argmin(axis=1)
            cluster_ids[dropped_nodes] = nearest.astype(np.int64)
        elif dropped_nodes.size > 0:
            cluster_ids[dropped_nodes] = pseudo_labels[dropped_nodes]

    return _cluster_stats_from_assignments(embeddings, cluster_ids, mode=mode)


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
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    cluster_ids, _, _ = gb_division_on_embeddings(
        embeddings_np, target_clusters, mode=mode
    )
    return torch.from_numpy(cluster_ids).long()


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
            centroids: torch.Tensor, optional - 预先计算的中心（用于 warm-start）

        返回:
            cluster_ids: torch.Tensor, shape (n,) - 簇ID
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = embeddings

        if centroids is not None and isinstance(centroids, torch.Tensor):
            centroids_np = centroids.detach().cpu().numpy()
        else:
            centroids_np = centroids

        cluster_ids, centers, radii = gb_division_on_embeddings(
            embeddings_np,
            self.n_clusters,
            mode=self.mode,
            centroids=centroids_np,
        )

        self.centroids = torch.from_numpy(centers).float()
        self.radii = torch.from_numpy(radii).float()
        return torch.from_numpy(cluster_ids).long()


class KMeansCluster:
    """KMeans 聚类器，提供与 GBCluster 一致的接口"""

    def __init__(
        self,
        n_clusters,
        mode="euclidean",
        verbose=0,
        max_iter=100,
        tol=1e-4,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.mode = mode
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.radii = None

    def _rng(self):
        if self.random_state is None:
            return np.random
        return np.random.RandomState(self.random_state)

    def _init_centers(self, embeddings, k, initial_centers=None):
        n, d = embeddings.shape
        centers = np.zeros((k, d), dtype=np.float32)
        start = 0

        if initial_centers is not None and len(initial_centers) > 0:
            initial_centers = np.asarray(initial_centers, dtype=np.float32)
            copy_len = min(k, initial_centers.shape[0])
            centers[:copy_len] = initial_centers[:copy_len]
            start = copy_len

        if start >= k:
            return centers

        rng = self._rng()
        if start == 0:
            centers[0] = embeddings[rng.randint(n)]
            start = 1

        for center_idx in range(start, k):
            min_dists = _compute_distances(
                embeddings, centers[:center_idx], mode=self.mode
            ).min(axis=1)
            probs = np.maximum(min_dists**2, 1e-12)
            prob_sum = probs.sum()

            if prob_sum <= 1e-12 or np.isnan(prob_sum) or np.isinf(prob_sum):
                next_center_idx = rng.randint(n)
            else:
                probs = probs / prob_sum
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    next_center_idx = rng.randint(n)
                else:
                    next_center_idx = rng.choice(n, p=probs)

            centers[center_idx] = embeddings[next_center_idx]

        return centers

    def fit_predict(self, embeddings, centroids=None):
        """
        对嵌入向量进行 KMeans 聚类。

        参数:
            embeddings: torch.Tensor or np.ndarray, shape (n, d)
            centroids: torch.Tensor or np.ndarray, optional - warm-start 中心

        返回:
            cluster_ids: torch.Tensor, shape (n,) - 簇ID
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = np.asarray(embeddings)

        embeddings_np = np.asarray(embeddings_np, dtype=np.float32)
        if embeddings_np.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if np.any(np.isnan(embeddings_np)) or np.any(np.isinf(embeddings_np)):
            embeddings_np = np.nan_to_num(
                embeddings_np, nan=0.0, posinf=0.0, neginf=0.0
            )

        n, d = embeddings_np.shape
        k = min(max(int(self.n_clusters), 0), n)
        if n == 0 or k <= 0:
            self.centroids = torch.zeros((0, d), dtype=torch.float32)
            self.radii = torch.zeros(0, dtype=torch.float32)
            return torch.zeros(n, dtype=torch.long)

        if centroids is not None and isinstance(centroids, torch.Tensor):
            centroids_np = centroids.detach().cpu().numpy()
        elif centroids is not None:
            centroids_np = np.asarray(centroids)
        else:
            centroids_np = None

        if (
            centroids_np is not None
            and centroids_np.ndim == 2
            and centroids_np.shape[1] == d
        ):
            centers = self._init_centers(embeddings_np, k, initial_centers=centroids_np)
        else:
            centers = self._init_centers(embeddings_np, k)

        cluster_ids = np.zeros(n, dtype=np.int64)
        for _ in range(self.max_iter):
            dists = _compute_distances(embeddings_np, centers, mode=self.mode)
            cluster_ids = dists.argmin(axis=1).astype(np.int64)

            new_centers = centers.copy()
            for cid in range(k):
                points = embeddings_np[cluster_ids == cid]
                if points.size > 0:
                    new_centers[cid] = points.mean(axis=0)

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift <= self.tol:
                break

        radii = np.zeros(k, dtype=np.float32)
        for cid in range(k):
            points = embeddings_np[cluster_ids == cid]
            radii[cid] = _compute_radius(points, centers[cid], mode=self.mode)

        self.centroids = torch.from_numpy(centers).float()
        self.radii = torch.from_numpy(radii).float()
        return torch.from_numpy(cluster_ids).long()
