import numpy as np
from tools.find_major import find_major
from tools.find_major_num import find_major_num
import networkx as nx


def split_2_co(graph, id_dict, Ci, index, total_degree_dict):
    data = Ci[0]  # 获取当前粒球中的节点数据
    degree_dict = Ci[1]  # 获取当前粒球中的度数字典
    ball_1 = []  # 初始化第一个子簇
    ball_2 = []  # 初始化第二个子簇
    subnodes = []  # 用于存储当前粒球的节点 ID
    for node in data:
        subnodes.append(node[0])
    subgraph = graph.subgraph(subnodes)
    Distance_1 = nx.single_source_shortest_path_length(subgraph, id_dict[index[0]])
    Distance_2 = nx.single_source_shortest_path_length(subgraph, id_dict[index[1]])

    for new_id in degree_dict:
        if Distance_1[id_dict[new_id]] <= Distance_2[id_dict[new_id]]:
            ball_1.append(new_id)
        else:
            ball_2.append(new_id)

    ball_1 = np.array(ball_1).astype(int)
    ball_2 = np.array(ball_2).astype(int)
    data1 = []  # 存储 ball_1 中的节点数据
    data2 = []  # 存储 ball_2 中的节点数据
    for i in ball_1:
        for j in data:
            if j[-1] == i:
                data1.append(j)
    for i in ball_2:
        for j in data:
            if j[-1] == i:
                data2.append(j)
    d1 = {}  # 存储 ball_1 中每个节点的度数
    d2 = {}  # 存储 ball_2 中每个节点的度数
    for i in range(len(ball_1)):
        d1.update({ball_1[i]: total_degree_dict[ball_1[i]]})
    for i in range(len(ball_2)):
        d2.update({ball_2[i]: total_degree_dict[ball_2[i]]})
    major_label1 = find_major(data1)
    major_label_num1, num_len1 = find_major_num(data1, major_label1)
    major_label2 = find_major(data2)
    major_label_num2, num_len2 = find_major_num(data2, major_label2)
    if num_len1 == 0:
        C1 = []
    else:
        C1 = [data1, d1, float(major_label_num1/num_len1)]
    if num_len2 == 0:
        C2 = []
    else:
        C2 = [data2, d2, float(major_label_num2/num_len2)]
    return C1, C2


def split_2_co_pagerank(graph, id_dict, GB, index, total_degree_dict):
    # 基于PageRank的分割逻辑
    original_id = id_dict[index]
    neighbors = list(graph.neighbors(original_id))
    
    # 获取PageRank值
    pr = nx.pagerank(graph)
    
    # 划分逻辑（示例实现）
    cluster1 = {
        'nodes': [node for node in GB['nodes'] if pr[id_dict[node[-1]]] >= pr[original_id]],
        'degrees': {},
        'kcore': 0
    }
    
    cluster2 = {
        'nodes': [node for node in GB['nodes'] if pr[id_dict[node[-1]]] < pr[original_id]],
        'degrees': {},
        'kcore': 0
    }
    
    # 更新度数字典
    for cluster in [cluster1, cluster2]:
        node_ids = [node[-1] for node in cluster['nodes']]
        cluster['degrees'] = {nid: total_degree_dict[nid] for nid in node_ids}
        
    return cluster1, cluster2