import json
import evaluation as eval_utils
from learn_from_query import est_AI1, est_AI2
from statistics import TableStats

# 文件路径
stats_json_file = './title_stats.json'
train_json_file = './query_train_18000.json'
test_json_file = './validation_2000.json'
columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']

# 加载表统计信息和数据
print("加载表统计信息和训练/测试数据...")
table_stats = TableStats.load_from_json_file(stats_json_file, columns)

with open(train_json_file, 'r') as f:
    train_data = json.load(f)
with open(test_json_file, 'r') as f:
    test_data = json.load(f)

# 生成预测结果
print("生成预测结果...")
_, _, est_AI1_test, act_test = est_AI1(
    train_data=train_data,
    test_data=test_data,
    table_stats=table_stats,
    columns=columns
)

_, _, est_AI2_test, _ = est_AI2(
    train_data=train_data,
    test_data=test_data,
    table_stats=table_stats,
    columns=columns
)

# 生成报告
print("生成报告...")
eval_utils.gen_report(
    act=act_test,
    est_results={
        "AI1": est_AI1_test,
        "AI2": est_AI2_test,
    }
)

print("报告已生成，保存在 ./eval/ 目录中。")
