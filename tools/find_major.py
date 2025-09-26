def find_major(nums):
    candidate = 0 # 初始化候选标签为 0
    count = 0 # 初始化计数器为 0

    # 投票算法（Boyer-Moore Majority Vote Algorithm） 来找到出现次数最多的标签
    for num in nums:
        if num[-2] != -1:
            if count == 0:
                candidate = num[-2]
                count = 1
            elif num[-2] == candidate:
                count += 1
            else:
                count -= 1

    return candidate