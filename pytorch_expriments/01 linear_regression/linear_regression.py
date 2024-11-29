# -- 我们要用梯度下降来优化线性回归模型的参数 假设我们有一组数据点，我们希望找到一条直线 y = wx + b
# -- 使得所有数据点尽可能接近这条直线。损失函数为MSE，使用梯度下降来更新参数

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(0)

# 1. 生成训练数据: y = 2x + 1 + 噪声
X = 3 * np.random.rand(100, 1)  # X属于[0, 3]
y = 2 * X + 1 + 0.5 * np.random.randn(100, 1)  # 真实关系: y = 2x + 1

# 2. 定义线性模型
def forward(X, w, b):  # 2 usages
    return X * w + b

# 3. 定义损失函数（均方误差）
def cost(y, y_pred):  # 1 usage
    return ((y - y_pred) ** 2).mean()

# 4. 定义梯度下降函数
def gradient_descent(X, y, w, b, learning_rate, epochs):  # 1 usage
    n = len(y)  # 用于记录损失值和梯度的更新
    losses = []
    w_values = []
    for epoch in range(epochs):
        # 预测
        y_pred = forward(X, w, b)
        # 计算损失
        loss = cost(y, y_pred)  # 记录损失值和梯度的更新过程
        losses.append(loss)
        w_values.append(w)
        # 计算梯度：损失函数(MSE)对权值(w)、偏置(b)求导
        dw = -(2 / n) * np.sum(X * (y - y_pred))
        db = -(2 / n) * np.sum(y - y_pred)
        # 梯度下降更新参数
        w -= learning_rate * dw
        b -= learning_rate * db
        # 每100次输出当前的损失值
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    return w, b, losses, w_values

# 初始化参数
w_init = 0.0
b_init = 0.0
learning_rate = 0.01
epochs = 1000

# 运行梯度下降
w_final, b_final, losses, w_values = gradient_descent(X, y, w_init, b_init, learning_rate, epochs)

# 打印最终更新得到的参数
print(f"Final parameters: w = {w_final:.4f}, b = {b_final:.4f}")

# 绘制原始数据点
plt.scatter(X, y, color='blue', label='Data')

# 使用最终的w和b绘制拟合直线
y_pred = forward(X, w_final, b_final)
plt.plot(X, y_pred, color='red', label='Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 绘制损失值随epoch变化的曲线图
plt.figure(figsize=(12, 5))

# 损失的Loss变化曲线
plt.subplot(1, 2, 1)
plt.plot(range(epochs), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# w值更新变化曲线
plt.subplot(1, 2, 2)
plt.plot(range(epochs), w_values, label='w Value', color='orange')
plt.xlabel('Epoch')
plt.ylabel('w Value')
plt.title('w Value over Epochs')
plt.legend()
plt.show()