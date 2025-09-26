from tools.find_major import find_major

# 计算每个粒度中的主要标签，并添加到粒球信息中
def purification(C):
    new_clusters = []
    for cluster in C:
        data = cluster[0]
        D_dict = cluster[1]
        pur = cluster[2]
        major_label = find_major(data)
        temp_C = [data, D_dict, pur, major_label]
        new_clusters.append(temp_C)

    return new_clusters
