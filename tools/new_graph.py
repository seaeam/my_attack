import numpy as np
import networkx as nx

def new_graph(granular_ball_list, graph):
    length = len(granular_ball_list)  # 获取粒度列表的长度
    graph_new = nx.Graph()  # 创建一个新的空图对象
    nodes = np.array(graph.nodes())  # 获取原始图中的所有节点，并转换为 NumPy 数组

    # 将粒球作为一个节点添加到新图中
    for i in range(length):
        granular_ball = granular_ball_list[i]
        graph_new.add_node(i, label=granular_ball[-1])

    # 记录每个节点所属的粒球
    node_to_granular_ball_index = {}
    for i, granular_ball in enumerate(granular_ball_list):
        for node in granular_ball[0]:
            node_index = int(node[-1])
            node_to_granular_ball_index[nodes[node_index]] = i

    # 只有当 u 和 v 属于不同的粒球时，才在新图中添加边。这表示不同粒度之间存在某种连接关系。
    for u, v in graph.edges():
        if u in node_to_granular_ball_index and v in node_to_granular_ball_index:
            u_index = node_to_granular_ball_index[u]
            v_index = node_to_granular_ball_index[v]
            if u_index != v_index:
                graph_new.add_edge(u_index, v_index)

    return graph_new
