def del_outlier(graph):
    isolated_nodes = []

    for node in graph.nodes():
        if graph.degree(node) == 0:
            isolated_nodes.append(node)

    graph.remove_nodes_from(isolated_nodes)

    # 返回移除孤立节点后的图
    return graph