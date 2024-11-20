import json
from learn_from_query import est_AI1, est_AI2
from range_query import ParsedRangeQuery
from statistics import TableStats

# 文件路径
query_train_file = 'query_train_18000.json'  # 原始训练数据
stats_file = 'title_stats.json'  # 数据库统计信息文件
test_file = 'validation_2000.json'  # 测试数据集

# 定义要考虑的列
considered_cols = ['production_year', 'season_nr', 'episode_nr']

# Step 1: 加载数据
print("加载训练数据和统计信息...")
with open(query_train_file, 'r') as f:
    train_data = json.load(f)

with open(test_file, 'r') as f:
    test_data = json.load(f)

# 加载表统计信息（需要 TableStats 实现）
table_stats = TableStats.load_from_json_file(stats_file, considered_cols)

# Step 2: 使用 est_AI1 进行训练和评估
print("使用 est_AI1 进行训练和评估...")
train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_AI1(
    train_data=train_data,
    test_data=test_data,
    table_stats=table_stats,
    columns=considered_cols
)

# Step 3: 评估模型性能
print("评估 est_AI1 的模型性能...")
from sklearn.metrics import mean_squared_error

train_mse = mean_squared_error(train_act_rows, train_est_rows)
test_mse = mean_squared_error(test_act_rows, test_est_rows)
print(f"训练集均方误差 (MSE): {train_mse}")
print(f"测试集均方误差 (MSE): {test_mse}")

# Step 4: 使用 est_AI2 进行训练和评估
print("使用 est_AI2 进行训练和评估...")
train_est_rows2, train_act_rows2, test_est_rows2, test_act_rows2 = est_AI2(
    train_data=train_data,
    test_data=test_data,
    table_stats=table_stats,
    columns=considered_cols
)

# Step 5: 评估模型性能
print("评估 est_AI2 的模型性能...")
train_mse2 = mean_squared_error(train_act_rows2, train_est_rows2)
test_mse2 = mean_squared_error(test_act_rows2, test_est_rows2)
print(f"训练集均方误差 (MSE): {train_mse2}")
print(f"测试集均方误差 (MSE): {test_mse2}")
