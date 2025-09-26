from tools import *
import numpy as np
import networkx as nx
import matplotlib
from tools.add_noise import add_noise

matplotlib.use("TkAgg")
import scipy.sparse as sp
from tools.split_ball_purity import split_ball_purity
from tools.purification import purification
from tools.add_id import add_id
from tools.new_graph import new_graph
from tools.corse_split import initial_splite
from tools.add_purity import add_purity
from tools.split_ball_purity import split_ball_further


def gb_division(data, args, fea=None):
    seed = 0

    # 1) 可选加噪
    if getattr(args, "noise", 0) == 1:
        data = add_noise(data)

    C = []

    # 2) 基础数据准备（按你的原逻辑）
    data_file = data.x.numpy()  # [N, d]
    total_balls_num = int(args.ball_r * len(data_file))  # 目标粒球数

    # 邻接（COO）
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    n_nodes = int(data.y.shape[0])
    adjacency_matrix = sp.coo_matrix(
        (np.ones_like(row, dtype=np.float32), (row, col)),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    )

    # 测试集标签置 -1（避免泄露）
    for i in range(len(data.test_mask)):
        if bool(data.test_mask[i]):
            data.y[i] = -1
    data_labels = data.y.numpy()

    # 3) 构图 & 去孤点
    graph = get_dataset.get_dataset(data_file, adjacency_matrix, data_labels, seed)
    graph = del_outlier.del_outlier(graph)

    # 4) 取属性与标签
    node_attributes = nx.get_node_attributes(graph, "attributes")
    attributes = np.array(list(node_attributes.values()))
    node_labels = nx.get_node_attributes(graph, "label")
    labels = np.array(list(node_labels.values()))
    labels = np.reshape(labels, (nx.number_of_nodes(graph), 1))  # n×1

    # 5) 连续编号 & old/new 映射
    total_degree_dict_old = dict(graph.degree())  # {old_id: deg}
    total_degree_dict = {}
    for new_id, (_, deg) in enumerate(total_degree_dict_old.items()):
        total_degree_dict[new_id] = deg

    # new->old / old->new
    id_dict = {}  # new -> old
    id_dict_oldtonew = {}  # old -> new
    for new, old in enumerate(total_degree_dict_old):
        id_dict[new] = old
        id_dict_oldtonew[old] = new

    # 6) 组装数据矩阵（索引|特征|标签）并 add_id
    indices = np.array(list(total_degree_dict_old.keys())).reshape(
        -1, 1
    )  # old ids (as your original)
    data_mat = np.concatenate((indices, attributes, labels), axis=1)
    data_mat = add_id(data_mat)

    # 7) 初分
    C.append([data_mat, total_degree_dict])
    new_C = initial_splite(
        C, graph, id_dict, id_dict_oldtonew, labels, total_degree_dict
    )

    # 8) 控制粒球数量（裁剪）
    target = 1
    while len(new_C) > total_balls_num:
        cut_pos = 0
        new_C.sort(key=lambda x: len(x[0]))  # 小到大
        for i in range(len(new_C)):
            if len(new_C[i][0]) == target + 1:
                cut_pos = i
                break
        target += 1
        new_C = new_C[cut_pos:]

    # 9) 纯度 + 细分 + 进一步细分 + 净化
    new_C = add_purity(new_C)
    new_C = split_ball_purity(graph, id_dict, new_C, total_degree_dict, total_balls_num)
    if len(new_C) < total_balls_num:
        new_C = split_ball_further(
            graph, id_dict, new_C, total_degree_dict, total_balls_num
        )
    new_C = purification(new_C)

    # 10) 计算粒球特征（均值）
    GB_features = []
    for GB in new_C:
        arr = np.array(GB[0])
        # 原逻辑：去掉索引与末尾两个字段（保持和你原版一致）
        slice_ = arr[:, 1:-2]
        GB_features.append(slice_.mean(axis=0))
    GB_features = np.array(GB_features)

    # 11) 粒球图
    GB_graph = new_graph(new_C, graph)

    new_f = {}
    gb_labels = [GB[-1] for GB in new_C]
    new_f["gb_labels"] = np.array(gb_labels)

    C_adj = sp.coo_matrix(nx.to_numpy_array(GB_graph))
    new_f["adj"] = np.vstack((C_adj.row, C_adj.col)).astype(np.int64)
    new_f["gb_features"] = GB_features

    # 12) —— 关键改动：始终返回粒球映射（old/new 双版本 + 兼容键）——
    gb2nodes_old, gb2nodes_new = [], []
    node2gb_old, node2gb_new = {}, {}

    for gb_idx, GB in enumerate(new_C):
        rows = np.array(GB[0])
        # 你的实现：第 0 列是“old”编号
        nodes_old = [int(u) for u in rows[:, 0].tolist()]
        gb2nodes_old.append(nodes_old)
        for u in nodes_old:
            node2gb_old[int(u)] = int(gb_idx)
        # 映射到“new”连续编号
        nodes_new = []
        for u in nodes_old:
            if int(u) in id_dict_oldtonew:
                v = int(id_dict_oldtonew[int(u)])
                nodes_new.append(v)
                node2gb_new[v] = int(gb_idx)
        gb2nodes_new.append(nodes_new)

    # 双向映射
    new_f["old2new"] = {int(k): int(v) for k, v in id_dict_oldtonew.items()}
    new_f["new2old"] = {int(k): int(v) for k, v in id_dict.items()}

    # 两套节点表 + 两套映射
    new_f["gb2nodes_old"] = gb2nodes_old
    new_f["gb2nodes_new"] = gb2nodes_new
    new_f["node2gb_old"] = node2gb_old
    new_f["node2gb_new"] = node2gb_new

    # 为兼容你现有调用，保留这两个简化键（指向 old 版）
    new_f["gb2nodes"] = gb2nodes_old
    new_f["node2gb"] = node2gb_old

    # 13) 若传入 fea，则把每个节点特征替换为其所属粒球均值
    if fea is not None:
        # 尝试用 old 编号直接映射（与你原版一致）
        node_to_gb_map = {}
        for idx, GB in enumerate(new_C):
            nodes_in_gb = [int(node[0]) for node in GB[0]]
            for node in nodes_in_gb:
                node_to_gb_map[int(node)] = int(idx)

        # 构造更新特征
        # 这里假设 fea.shape[0] 与攻击侧使用的节点编号一致（0..N-1）
        out_dim = GB_features.shape[0] and GB_features.shape[1] or 0
        updated_fea = np.zeros((fea.shape[0], out_dim), dtype=np.float32)

        # 用 old 编号直接赋值；越界则跳过
        for node, gb_idx in node_to_gb_map.items():
            if 0 <= int(node) < updated_fea.shape[0]:
                updated_fea[int(node)] = GB_features[int(gb_idx)]

        fea = updated_fea

    return new_f, fea
