# -- 使用sklearn自带的加州房价数据集进行回归
# -- 加州房价数据集一共有20640个数据，每个数据包含8个属性：经度、纬度、房龄...

# 1. 导入必要的库
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# 2. 加载加州房价数据集
california = datasets.fetch_california_housing()
# print(california.keys())  # 数据集属性
# print(california.target_names)  # 标签名称（房价中位数）
# print(california.feature_names)  # 训练数据每一维所代表的意义

# 3. 划分训练集和测试集
x = california.data
y = california.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# -- 数据预处理：对特征数据进行标准化，以确保所有的特征在同一尺度！
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 4. 创建并训练多层感知器回归模型
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, random_state=42, learning_rate_init=0.009,
                              solver="adam", verbose=False)
mlp_regressor.fit(x_train_scaled, y_train)

# -- 可视化训练过程中损失值变化
plt.plot(mlp_regressor.loss_curve_)
plt.title('Loss Curve during Training')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 5. 模型预测
y_pred = mlp_regressor.predict(x_test_scaled)

# -- 进行模型评估
mse = mean_squared_error(y_test, y_pred)  # 均方误差，越低越好
r2 = r2_score(y_test, y_pred)  # 决定系数，越接近于1越好，说明模型拟合的越好

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# 6. 可视化预测结果与真实值的对比
plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # 绘制y=x线
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.grid(True)
plt.show()