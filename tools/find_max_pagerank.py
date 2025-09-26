import networkx as nx

def find_max_pagerank(graph, nodes):
    pr = nx.pagerank(graph)
    max_pr = -1
    max_node = None
    
    for node_data in nodes:
        original_id = node_data[-1]  # 假设最后一个元素是原始节点ID
        node_pr = pr.get(original_id, 0)
        if node_pr > max_pr:
            max_pr = node_pr
            max_node = original_id
            
    return max_pr, max_node