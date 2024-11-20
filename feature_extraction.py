import json
from range_query import ParsedRangeQuery
from learn_from_query import extract_features_from_query

# 文件路径
input_file = 'query_train_18000.json'  # 包含 query 和 act_rows 的原始数据
output_file = 'parsed_features.json'  # 输出的特征文件

# 定义需要提取特征的列
considered_cols = ['production_year', 'season_nr']

# 加载查询数据
print("加载查询数据...")
with open(input_file, 'r') as f:
    queries = json.load(f)

# 存储解析结果和特征
parsed_results = []

print("开始解析查询并提取特征...")
for query_data in queries:
    query = query_data['query']  # SQL 查询字符串
    act_rows = query_data['act_rows']  # 目标值（实际基数）

    # 使用 ParsedRangeQuery 解析查询
    parsed_query = ParsedRangeQuery.parse_range_query(query)

    # 提取特征
    features = extract_features_from_query(parsed_query, table_stats=None, considered_cols=considered_cols)

    # 将结果存储到列表
    parsed_results.append({
        "query": query,
        "features": features,
        "act_rows": act_rows  # 加入目标值
    })

# 将解析结果和特征保存到文件
print("保存解析结果和特征...")
with open(output_file, 'w') as f:
    json.dump(parsed_results, f, indent=4)

print(f"完成！结果保存在 {output_file}")
