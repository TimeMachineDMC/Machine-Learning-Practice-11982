import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

# 文件路径
input_file = 'parsed_features.json'

# Step 1: 加载数据
print("加载数据...")
with open(input_file, 'r') as f:
    data = json.load(f)

# 提取特征和目标值
X = []  # 特征向量
y = []  # 实际基数
for record in data:
    X.append(record['features'])
    y.append(record['act_rows'])

# 转换为 NumPy 数组
X = np.array(X)
y = np.array(y)

# Step 2: 数据划分
print("划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: 定义模型
print("定义模型...")
# 使用线性回归
linear_model = LinearRegression()

# 或者使用 PyTorch 构建一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 使用线性回归或神经网络模型
use_nn = True  # 是否使用神经网络
if use_nn:
    model = SimpleNN(input_size=X_train.shape[1])
else:
    model = linear_model

# Step 4: 训练模型
if use_nn:
    # 转换数据为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("开始训练神经网络...")
    for epoch in range(100):  # 训练 100 个 epoch
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        # 每 10 个 epoch 输出一次验证集误差
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor).item()
            print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")
else:
    print("开始训练线性回归模型...")
    model.fit(X_train, y_train)

# Step 5: 验证模型
print("评估模型性能...")
if use_nn:
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor).numpy()
        val_loss = mean_squared_error(y_val, val_predictions)
else:
    val_predictions = model.predict(X_val)
    val_loss = mean_squared_error(y_val, val_predictions)

print(f"验证集均方误差: {val_loss}")

# 保存模型（仅对神经网络模型）
if use_nn:
    torch.save(model.state_dict(), 'trained_model.pth')
    print("神经网络模型已保存为 'trained_model.pth'")
