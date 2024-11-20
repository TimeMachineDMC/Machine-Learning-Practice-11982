import json
import random

# 文件路径
input_file = 'query_train_18000.json'  # 输入文件
train_output_file = 'train_filtered.json'  # 训练集输出文件
test_output_file = 'you_can_make_it.json'  # 测试集输出文件

# 划分比例
test_ratio = 0.1  # 10% 作为测试集，剩余90%作为训练集

# Step 1: 加载数据
print("加载数据中...")
with open(input_file, 'r') as f:
    data = json.load(f)

# Step 2: 确保数据总量
total_queries = len(data)  # 总数据条数
print(f"加载完成，共有 {total_queries} 条数据！")

# Step 3: 计算划分数量
test_size = int(total_queries * test_ratio)  # 测试集大小
train_size = total_queries - test_size  # 训练集大小
print(f"计划划分出 {test_size} 条数据作为测试集，剩余 {train_size} 条数据作为训练集...")

# Step 4: 随机划分数据
print("开始随机划分数据...")
test_data = random.sample(data, test_size)  # 从数据中随机抽取 test_size 条数据作为测试集
train_data = [query for query in data if query not in test_data]  # 剩余部分为训练集
print(f"划分完成！训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

# Step 5: 保存训练集和测试集到文件
print("保存数据中...")
with open(train_output_file, 'w') as f:  # 保存训练集
    json.dump(train_data, f, indent=4)  # 格式化保存（缩进4个空格）
with open(test_output_file, 'w') as f:  # 保存测试集
    json.dump(test_data, f, indent=4)
print(f"保存完成！训练集保存在 '{train_output_file}'，测试集保存在 '{test_output_file}'")

