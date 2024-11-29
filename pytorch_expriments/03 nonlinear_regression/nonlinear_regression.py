# -- 比较带激活函数和不带激活函数的神经网络拟合效果
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(0)

# 1. 生成非线性数据集: y = 2 * x^2 + 3 * x + 1 + 噪声
X = 2 * torch.rand(200, 1) - 1  # X ∈ [-1, 1] 范围内
y = 2 * X ** 2 + 3 * X + 1 + torch.randn(200, 1) * 0.1  # 添加噪声

# 2. 模型设计
# 不带激活函数的网络
class LinearNet(nn.Module):  # 2 usages
    def __init__(self):
        super(LinearNet, self).__init__()
        self.hidden1 = nn.Linear(in_features=1, out_features=10)
        self.hidden2 = nn.Linear(in_features=10, out_features=5)
        self.output = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x

# 带激活函数的网络
class NonLinearNet(nn.Module):  # 2 usages
    def __init__(self):
        super(NonLinearNet, self).__init__()
        self.hidden1 = nn.Linear(in_features=1, out_features=10)
        self.hidden2 = nn.Linear(in_features=10, out_features=5)
        self.output = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))  # 使用ReLU激活函数
        x = torch.relu(self.hidden2(x))  # 使用ReLU激活函数
        x = self.output(x)
        return x

# 初始化模型
linear_model = LinearNet()
nonlinear_model = NonLinearNet()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer_linear = optim.SGD(linear_model.parameters(), lr=0.1)
optimizer_nonlinear = optim.SGD(nonlinear_model.parameters(), lr=0.1)

# 补充超参数
epochs = 1000           # 迭代世代数
linear_losses = []     # 线性损失值
nonlinear_losses = []  # 非线性损失值

# 4. 模型训练
# -- 训练不带激活函数的网络
for epoch in range(epochs):
    # 前向传播
    y_pred_linear = linear_model(X)
    # 计算损失
    loss_linear = criterion(y_pred_linear, y)
    linear_losses.append(loss_linear.item())
    # 反向传播和参数更新
    optimizer_linear.zero_grad()
    loss_linear.backward()
    optimizer_linear.step()

# -- 训练带激活函数的网络
for epoch in range(epochs):
    # 前向传播
    y_pred_nonlinear = nonlinear_model(X)
    # 计算损失
    loss_nonlinear = criterion(y_pred_nonlinear, y)
    nonlinear_losses.append(loss_nonlinear.item())
    # 反向传播和参数更新
    optimizer_nonlinear.zero_grad()
    loss_nonlinear.backward()
    optimizer_nonlinear.step()

# 5. 模型评估：在训练结束后评估模型的MSE（预测值与真实值之间的均方误差）
with torch.no_grad():
    # 计算不带激活函数模型的MSE
    final_y_pred_linear = linear_model(X)
    mse_linear = criterion(final_y_pred_linear, y).item()
    print(f"MSE of Linear Network (no activation): {mse_linear:.4f}")

    # 计算带激活函数模型的MSE
    final_y_pred_nonlinear = nonlinear_model(X)
    mse_nonlinear = criterion(final_y_pred_nonlinear, y).item()
    print(f"MSE of Nonlinear Network (with activation): {mse_nonlinear:.4f}")

# 绘制损失值变化曲线
plt.figure(figsize=(12, 5))

# -- 不带激活函数的损失值变化曲线
plt.subplot(1, 2, 1)
plt.plot(range(epochs), linear_losses, label='Linear Network Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs (Linear Network)')
plt.legend()

# -- 带激活函数的损失值变化曲线
plt.subplot(1, 2, 2)
plt.plot(range(epochs), nonlinear_losses, label='Nonlinear Network Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs (Nonlinear Network)')
plt.legend()
plt.show()

# 绘制拟合效果对比图
plt.figure(figsize=(12, 5))
plt.scatter(X.numpy(), y.numpy(), color='blue', label='Data')
# 使用不带激活函数的模型预测
plt.scatter(X.numpy(), final_y_pred_linear.numpy(), color='red', label='Linear Network Fit')
# 使用带激活函数的模型预测
plt.scatter(X.numpy(), final_y_pred_nonlinear.numpy(), color='green', label='Nonlinear Network Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Comparison of Linear and Nonlinear Network Fits')
plt.show()