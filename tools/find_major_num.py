def find_major_num(nums, target):
    sum = 0 # 初始化目标标签的计数器
    num_len = 0 # 初始化有效节点的计数器
    for num in nums:
        if num[-2] == target:
            sum += 1
        if num[-2] != -1:
            num_len += 1

    return sum, num_len
