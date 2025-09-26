import numpy as np
import networkx as nx


def get_dataset(data_points, adjacency_matrix, data_labels, seed=None):
    graph = nx.Graph()

    # 添加节点到图中
    for node_id, point in enumerate(data_points):
        graph.add_node(node_id, attributes=point)

    # 为节点添加标签
    for node_id, label in enumerate(data_labels):
        graph.nodes[node_id]['label'] = label

    adjacency_matrix = adjacency_matrix.tocsr()

    # 获取图中所有的边
    rows, cols = adjacency_matrix.nonzero()

    # 添加边到图中
    for i, j in zip(rows, cols):
        # 排除自环
        if i != j:
            graph.add_edge(i, j, edge_attr=np.zeros(3)) #初始化边属性

    return graph



