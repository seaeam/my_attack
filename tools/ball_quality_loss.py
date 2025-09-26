import torch
import networkx as nx
import torch.nn.functional as F

# 计算同配比例损失（Homophily Loss）
def homophily_loss(output, adj, labels, device):
    cosine_sim = F.cosine_similarity(output.unsqueeze(0), output.unsqueeze(1), dim=2)
    mask = adj.bool()
    sim_masked = cosine_sim * mask.float()
    same_class_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    # 同类节点相似度较大，不同类节点相似度较小
    loss = F.mse_loss(sim_masked * same_class_mask, torch.ones_like(sim_masked) * same_class_mask)
    return loss

# 计算k-core损失（K-Core Loss）
def kcore_loss(original_graph, new_graph, device):
    original_nx_graph = nx.from_scipy_sparse_matrix(original_graph.to_sparse().cpu())
    new_nx_graph = nx.from_scipy_sparse_matrix(new_graph.to_sparse().cpu())

    original_kcore = nx.core_number(original_nx_graph)
    new_kcore = nx.core_number(new_nx_graph)

    original_kcore_values = torch.tensor(list(original_kcore.values()), dtype=torch.float32, device=device)
    new_kcore_values = torch.tensor(list(new_kcore.values()), dtype=torch.float32, device=device)

    # L2损失，计算k-core值差异
    loss = F.mse_loss(original_kcore_values, new_kcore_values)
    return loss

# 总粒球质量评估损失：同配比例 + k-core损失
def ball_quality_loss(output, adj, labels, original_graph, new_graph, device, lambda_kcore=0.1):
    homophily = homophily_loss(output, adj, labels, device)
    kcore = kcore_loss(original_graph, new_graph, device)

    total_loss = homophily + lambda_kcore * kcore
    return total_loss
