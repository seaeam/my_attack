from tools.find_max_pagerank import find_max_pagerank  # 新增导入
from tools.split_2_co import split_2_co_pagerank  # 修改导入
import networkx as nx

def split_ball_purity(graph, id_dict, C, total_degree_dict, total_balls_num, kcore_threshold=2):
    cur_ball_num = len(C)
    while True:
        # 按k-core值升序排序
        C.sort(key=lambda x: x['kcore'])  # 修改排序依据
        
        if cur_ball_num >= total_balls_num or C[0]['kcore'] >= kcore_threshold:
            break
            
        GB = C.pop(0)
        # 找PageRank最大的节点
        value, index = find_max_pagerank(graph, GB['nodes'])  # 修改调用
        
        # 使用PageRank分割
        cluster1, cluster2 = split_2_co_pagerank(graph, id_dict, GB, index, total_degree_dict)
        
        # 更新k-core值
        for cluster in [cluster1, cluster2]:
            if cluster['nodes']:
                sub_g = graph.subgraph([id_dict[node[-1]] for node in cluster['nodes']])
                core_number = nx.core_number(sub_g)
                cluster['kcore'] = sum(core_number.values()) / len(core_number)
                C.append(cluster)
                
        cur_ball_num += 1
        
    return C

def split_ball_further(graph, id_dict, C, total_degree_dict, total_balls_num, kcore_threshold=2):
    cur_ball_num = len(C)
    while cur_ball_num < total_balls_num:
        # 按节点数量降序排序
        C.sort(key=lambda x: len(x['nodes']), reverse=True)
        GB = C.pop(0)
        
        # 找PageRank最大的两个节点
        pr_values = [(node[-1], nx.pagerank(graph)[id_dict[node[-1]]]) for node in GB['nodes']]
        pr_values.sort(key=lambda x: x[1], reverse=True)
        indices = [node[0] for node in pr_values[:2]]
        
        # 双重分割
        for index in indices:
            cluster1, cluster2 = split_2_co_pagerank(graph, id_dict, GB, index, total_degree_dict)
            
            for cluster in [cluster1, cluster2]:
                if cluster['nodes']:
                    sub_g = graph.subgraph([id_dict[node[-1]] for node in cluster['nodes']])
                    core_number = nx.core_number(sub_g)
                    cluster['kcore'] = sum(core_number.values()) / len(core_number)
                    C.append(cluster)
                    
            cur_ball_num += 1
            
    return C[:total_balls_num]