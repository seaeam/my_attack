from tools.find_max_degree import find_max_degree
from tools.split_2_co import split_2_co
ini = float('inf')
current_balls_num = 0


def split_ball_purity(graph, id_dict, C, total_degree_dict, total_balls_num, purity_threshold=1):
    cur_ball_num = len(C) # 当前粒球数量
    while True:

        C.sort(key=lambda x: x[-1]) #纯度小的排在前面，优先处理
        # 如果当前粒球数已经达到了目标粒球数，或者所有粒球的纯度已经达到了设定的阈值，则退出循环
        if cur_ball_num >= total_balls_num or C[0][-1] >= purity_threshold:
            break

        GB = C.pop(0) # GB 代表当前正在处理的粒球
        value, index = find_max_degree(GB[1]) # 找出当前粒球中度数最大的节点的索引
        cluster1, cluster2 = split_2_co(graph, id_dict, GB, index, total_degree_dict) # 细化粒球，分为两个更小的子球
        temp_num = -1 # 记录当前分割操作后新增加的粒度数量
        if len(cluster1) != 0:
            C.append(cluster1)
            temp_num += 1
        if len(cluster2) != 0:
            C.append(cluster2)
            temp_num += 1
        cur_ball_num += temp_num
    return C


def split_ball_further(graph, id_dict, C, total_degree_dict, total_balls_num,purity_threshold=1.0):
    cur_ball_num = len(C)
    while True:
        if cur_ball_num >= total_balls_num:
            break
        C.sort(key=lambda x: len(x[0]), reverse=True)
        GB = C.pop(0)
        value, index = find_max_degree(GB[1])
        cluster1, cluster2 = split_2_co(graph, id_dict, GB, index, total_degree_dict)
        temp_num = -1
        if len(cluster1) != 0:
            C.append(cluster1)
            temp_num += 1
        if len(cluster2) != 0:
            C.append(cluster2)
            temp_num += 1
        cur_ball_num += temp_num
    return C

