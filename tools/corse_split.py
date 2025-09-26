import numpy as np
from math import sqrt
ini = float('inf')
import networkx as nx

def initial_splite(C, graph, id_dict, id_dict_oldtonew, labels, total_degree_dict):

    # 计算图中的连通分量
    connected_components = list(nx.connected_components(graph))
    new_clusters = []
    for i, component in enumerate(connected_components, start=1):

        # 创建一个子图 subgraph，包含该连通分量的节点和边
        subgraph = graph.subgraph(list(component))
        new_node_ids = []
        for old_id in list(component):
            new_node_ids.append(id_dict_oldtonew[old_id])

        component_data = [C[0][0][i] for i in new_node_ids]
        component_degree_dict = {node: total_degree_dict[node] for node in new_node_ids}
        component_C = [component_data, component_degree_dict]

        # 选择质心
        centers = select_initial_centers_using_degree(component_C, component_degree_dict)

        if len(centers) != 0:
            # 计算中心节点到其他所有节点的最短路径长度
            Distances = []
            for center in centers:
                Distance = nx.single_source_shortest_path_length(subgraph, id_dict[center])
                Distances.append(Distance)

            balls_data_nodes = [[] for _ in centers]
            balls_degree_dict = [{} for _ in centers]
            # 对于当前连通分量中的每个节点 old_id
            for old_id in list(component):

                min = np.inf # 最小距离
                min_center_idx = -1 # 最小距离对应的中心节点的 ID

                for Distance in Distances:
                    if Distance[old_id] < min:
                        min = Distance[old_id]
                        min_center_idx = next(iter(Distance.keys()))
                center_index = centers.index(id_dict_oldtonew[min_center_idx])
                balls_data_nodes[center_index].append(C[0][0][id_dict_oldtonew[old_id]])

            for index, center in enumerate(centers):
                ball_nodes = balls_data_nodes[index]
                ball_nodes_index = [int(row[-1]) for row in ball_nodes] # 提取归属该中心的所有节点的 ID

                for node in ball_nodes_index: #对于归属当前中心的每个节点 node，添加度数信息
                    balls_degree_dict[index][node] = total_degree_dict[node]

            balls = [ # 将每个“球”中的节点数据和节点度数字典打包在一起
                [balls_data_nodes[i], balls_degree_dict[i]]
                for i in range(len(centers))
            ]
            for ball in balls:
                new_clusters.append(ball)
    return new_clusters

def select_initial_centers_using_degree(component_C, degree_dict):
    node_info = component_C[0]
    class_nodes_dict = {}
    # 标签到节点 ID 列表的映射
    for info in node_info:
        label = info[-2]
        node_index = int(info[-1])
        if label not in class_nodes_dict and label != -1:
            class_nodes_dict[label] = []
        if label != -1:
            class_nodes_dict[label].append(node_index)

    num_classes = len(class_nodes_dict)
    num_nodes = len(node_info)

    if num_classes != 0:
        centers_per_class = max(1, int(sqrt(num_nodes) / num_classes))

    centers = []
    # 每个标签选centers_per_class个度数较大的节点为质心
    for label, nodes in class_nodes_dict.items():
        degrees = [(node, degree_dict[node]) for node in nodes if node in degree_dict]
        degrees.sort(key=lambda x: x[1], reverse=True)
        selected_centers = [node for node, _ in degrees[:centers_per_class]]
        centers.extend(selected_centers)
    return centers


# pagerank进行粗粒度划分
import numpy as np
from math import sqrt
import networkx as nx

def initial_splite_pagerank(C, graph, id_dict, id_dict_oldtonew, labels, total_degree_dict):
    connected_components = list(nx.connected_components(graph))
    new_clusters = []
    
    for component in connected_components:
        subgraph = graph.subgraph(list(component))
        new_node_ids = [id_dict_oldtonew[old_id] for old_id in component]
        
        component_data = [C[0][0][i] for i in new_node_ids]
        component_degree_dict = {node: total_degree_dict[node] for node in new_node_ids}
        component_C = [component_data, component_degree_dict]

        # 使用PageRank选择初始中心
        centers = select_initial_centers_using_pagerank(component_C, subgraph, id_dict_oldtonew)

        if centers:
            # 计算PageRank值
            pagerank = nx.pagerank(subgraph)
            
            balls_data_nodes = [[] for _ in centers]
            balls_degree_dict = [{} for _ in centers]
            
            # 分配节点到最近的PageRank中心
            for old_id in component:
                new_id = id_dict_oldtonew[old_id]
                max_pr = -1
                selected_center = None
                
                # 找到最近（PageRank最大）的中心
                for center in centers:
                    pr = pagerank.get(id_dict[center], 0)
                    if pr > max_pr:
                        max_pr = pr
                        selected_center = center
                
                center_index = centers.index(selected_center)
                balls_data_nodes[center_index].append(C[0][0][new_id])

            # 构建新的粒球
            for index in range(len(centers)):
                ball_nodes = balls_data_nodes[index]
                if not ball_nodes:
                    continue
                
                ball_nodes_index = [int(row[-1]) for row in ball_nodes]
                degree_dict = {node: total_degree_dict[node] for node in ball_nodes_index}
                
                # 计算k-core值
                sub_g = subgraph.subgraph([id_dict[node] for node in ball_nodes_index])
                core_number = nx.core_number(sub_g)
                avg_core = sum(core_number.values()) / len(core_number) if core_number else 0
                
                new_clusters.append({
                    'nodes': ball_nodes,
                    'degrees': degree_dict,
                    'kcore': avg_core
                })
    
    return new_clusters

def select_initial_centers_using_pagerank(component_C, subgraph, id_dict_oldtonew):
    pagerank = nx.pagerank(subgraph)
    node_pr = [(node, pagerank[node]) for node in subgraph.nodes()]
    node_pr.sort(key=lambda x: x[1], reverse=True)
    
    # 选择前sqrt(n)个节点作为中心
    n = len(node_pr)
    k = max(1, int(np.sqrt(n)))
    centers = [id_dict_oldtonew[node_pr[i][0]] for i in range(k)]
    
    return centers

