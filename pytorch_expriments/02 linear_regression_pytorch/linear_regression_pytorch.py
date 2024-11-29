# -- 使用pytorch反向传播自动求导实现一元线性回归
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(0)

# 1. 准备训练数据
# -- 生成数据 y = 2x + 1 + 噪声
# -- 因torch.rand生成的数据为Tensor数据类型，所有X,y均为Tensor类型
X = 3 * torch.rand(100, 1)
y = 1 + 2 * X + 0.5 * torch.randn(100, 1)  # 实际需要拟合的线性回归模型 y = 2x + 1

# 2. 设计一元线性模型
class LinearModel(nn.Module):  # 2 usages
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1, bias=True)
        # 将权重和偏置初始化为0
        nn.init.constant_(self.linear.weight, val=0.0)
        nn.init.constant_(self.linear.bias, val=0.0)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# 实例化一个模型
model = LinearModel()

# 3. 定义损失函数（均方误差）和优化器（随机梯度下降）
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 初始化2个列表用于存储每个epoch的损失值和权重更新值
losses = []
w_values = []
# 设置训练次数（多少轮次）
epochs = 1000

# 4. 训练模型
for epoch in range(epochs):
    # 前向传播：计算模型的预测值
    y_pred = model(X)
    # 计算损失
    loss = criterion(y_pred, y)
    # 记录当前的损失值和权重
    losses.append(loss.item())
    w_values.append(model.linear.weight.item())
    # 反向传播：计算梯度
    optimizer.zero_grad()  # 清零梯度：每次反向传播都会计算梯度，清零是防止累加
    loss.backward()  # 计算梯度
    # 更新参数：梯度下降法
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 打印最终更新的权重和参数
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# 绘制原始数据点
plt.scatter(X.detach().numpy(), y.detach().numpy(), color='blue', label='Data')

# 使用模型进行预测，绘制拟合直线
with torch.no_grad():
    y_pred = model(X)
plt.plot(X.detach().numpy(), y_pred.detach().numpy(), color='red', label='Fitted Line')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()