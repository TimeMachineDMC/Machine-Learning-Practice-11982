from statistics import AVIEstimator, ExpBackoffEstimator, MinSelEstimator
import matplotlib.pyplot as plt

# 修复后的 Mock 类定义
class MockColumnStats:
    def __init__(self, min_value, max_value, total_rows, histogram=None):
        self.min_value = min_value  # 修改属性名称
        self.max_value = max_value
        self.total_rows = total_rows
        self.histogram = histogram or {}

    def min_val(self):
        return self.min_value  # 返回属性值

    def max_val(self):
        return self.max_value  # 返回属性值

    def between_row_count(self, left, right):
        count = 0
        for key, value in self.histogram.items():
            if left <= key < right:
                count += value
        return count


class MockTableStats:
    def __init__(self, total_rows, column_stats):
        self.row_count = total_rows
        self.columns = column_stats


class MockRangeQuery:
    def __init__(self, ranges):
        self.ranges = ranges

    def column_names(self):
        return list(self.ranges.keys())

    def column_range(self, col_name, min_val, max_val):
        return self.ranges.get(col_name, (min_val, max_val))


# 构造数据
column_stats = {
    "A": MockColumnStats(min_value=0, max_value=100, total_rows=1000, histogram={i: 10 for i in range(101)}),
    "B": MockColumnStats(min_value=0, max_value=50, total_rows=1000, histogram={i: 20 for i in range(51)})
}
table_stats = MockTableStats(total_rows=1000, column_stats=column_stats)
range_query = MockRangeQuery({"A": (20, 50), "B": (10, 30)})

# 估计
avi_selectivity = AVIEstimator.estimate(range_query, table_stats)
avi_cardinality = avi_selectivity * table_stats.row_count
ebo_selectivity = ExpBackoffEstimator.estimate(range_query, table_stats)
ebo_cardinality = ebo_selectivity * table_stats.row_count
minsel_selectivity = MinSelEstimator.estimate(range_query, table_stats)
minsel_cardinality = minsel_selectivity * table_stats.row_count

# 输出结果
print(f"AVI: {avi_cardinality}, EBO: {ebo_cardinality}, MinSel: {minsel_cardinality}")

# 绘图
methods = ['AVI', 'EBO', 'MinSel']
cardinalities = [avi_cardinality, ebo_cardinality, minsel_cardinality]
plt.bar(methods, cardinalities, color=['blue', 'green', 'red'])
plt.xlabel("Methods")
plt.ylabel("Estimated Cardinality")
plt.title("Comparison of Cardinality Estimation Methods")
plt.show()
