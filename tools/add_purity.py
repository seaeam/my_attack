from tools.find_major import find_major
from tools.find_major_num import find_major_num


def add_purity(C):
    for ball in C:
        nodes = ball[0] # 获取当前粒球中的节点列表
        major_label = find_major(nodes) # 出现频率最高的标签
        major_label_num, num_len = find_major_num(nodes, major_label) # 计算主要标签的数量和该粒度节点的总数
        ball.append(float(major_label_num/num_len)) # 计算纯度，并将其添加到粒球数据中

    return C # C=[[],{},纯度值]