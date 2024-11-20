import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
import torch
import torch.nn as nn
import torch.optim as optim
import statistics as stats


def min_max_normalize(v, min_v, max_v):
    assert max_v > min_v
    return (v - min_v) / (max_v - min_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    feature = []
    for col in considered_cols:
        if col in range_query.col_left:
            min_val = range_query.col_left[col]
            max_val = range_query.col_right[col]
            feature.extend([min_val, max_val])
        else:
            feature.extend([0, 0])
    return feature


def preprocess_queries(queries, table_stats, columns):
    features, labels = [], []
    for item in queries:
        query, act_rows = item['query'], item['act_rows']
        range_query = rq.ParsedRangeQuery.parse_range_query(query)
        feature = extract_features_from_query(range_query, table_stats, columns)
        features.append(feature)
        labels.append(act_rows)
    return features, labels


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        features, labels = preprocess_queries(queries, table_stats, columns)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def est_AI1(train_data, test_data, table_stats, columns):
    train_dataset = QueryDataset(train_data, table_stats, columns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = len(columns) * 2  # 每列包含 min 和 max 两个特征
    model = SimpleNN(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(20):  # 训练 20 个 epoch
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions.view(-1), labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # 测试模型
    train_est_rows, train_act_rows = [], []
    test_est_rows, test_act_rows = [], []

    model.eval()
    with torch.no_grad():
        for features, labels in train_loader:
            predictions = model(features).view(-1).tolist()
            train_est_rows.extend(predictions)
            train_act_rows.extend(labels.tolist())

        for features, labels in test_loader:
            predictions = model(features).view(-1).tolist()
            test_est_rows.extend(predictions)
            test_act_rows.extend(labels.tolist())

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def est_AI2(train_data, test_data, table_stats, columns):
    # 逻辑与 est_AI1 类似，可以定义不同的网络或优化方式
    return est_AI1(train_data, test_data, table_stats, columns)


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'ai1':
        est_fn = est_AI1
    else:
        est_fn = est_AI2

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

    name = f'{model}_train_{len(train_data)}'
    eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


if __name__ == '__main__':
    stats_json_file = './title_stats.json'
    train_json_file = './query_train_18000.json'
    test_json_file = './validation_2000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']

    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    eval_model('ai1', train_data, test_data, table_stats, columns)
    eval_model('ai2', train_data, test_data, table_stats, columns)
