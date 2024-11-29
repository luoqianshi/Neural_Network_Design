import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, log_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from datetime import datetime

# -------这里是超参数和全局变量定义------- #
# 数据预处理方法选择
# # data_preprocessing = 0 表示不做任何处理，以原始数据训练
# # data_preprocessing = 1 表示使用的方法是标准化处理
# # data_preprocessing = 2 表示使用的方法是最小最大缩放
data_preprocessing = 0
# 训练的总世代数
total_epoch = 2000
# 初始学习率
lr = 0.001
# batch_size 默认值为 'auto'，即 min(200, n_samples)
batch_size = 'auto'
# MLP模型隐藏层的网络形状
hidden_layer_sizes = (100,)
# 激活函数
activation = 'logistic'
# 优化器（求解器）
solver = 'adam'
# 模型权重参数暖启动
# 当 warm_start=True 时，模型在每次调用 fit 方法时不会重新初始化参数，
# 而是继续使用上一次训练的参数。这意味着你可以在多次调用 fit 方法时累积训练效果。
warm_start=True
# -------这里是超参数和全局变量定义------- #

# -------实验结果保存相关------- #
# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
# 创建结果目录
results_dir = f"./results/{current_time}"
os.makedirs(results_dir, exist_ok=True)
# -------实验结果保存相关------- #

# 加载数据
df = pd.read_csv("./data/Student_Performance.csv")

# 将'Extracurricular Activities'列映射为数值
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# 定义特征和目标变量
X = df.drop(columns="Performance Index")
y = df["Performance Index"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------数据预处理不同方法选择[这里是回归实验讨论的重点]------------------- #
# 1. 对特征数据进行标准化，以确保所有的特征在同一尺度！
if data_preprocessing == 0:
    x_train_processed = X_train
    x_test_processed = X_test
elif data_preprocessing == 1:
    scaler = StandardScaler()
    # 使用标准化，减均值，除方差
    x_train_processed = scaler.fit_transform(X_train)
    x_test_processed = scaler.transform(X_test)
elif data_preprocessing == 2:
    # 使用最大最小缩放，把数值缩放到[0,1]
    scaler = MinMaxScaler()
    x_train_processed = scaler.fit_transform(X_train)
    x_test_processed = scaler.transform(X_test)

# 将预处理后的数据转换回DataFrame，保留列名,同时在训练的时候也使用保留列名的DataFrame数据，方便后续分析。
x_train_processed_df = pd.DataFrame(x_train_processed, columns=X_train.columns)
x_test_processed_df = pd.DataFrame(x_test_processed, columns=X_train.columns)
# -------------------数据预处理不同方法选择[这里是回归实验讨论的重点]------------------- #


# 初始化MLPRegressor
model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver=solver,
    max_iter=1,
    random_state=42,
    warm_start=warm_start,
    learning_rate_init=lr,
    batch_size=batch_size,
    verbose=False)

# 使用tqdm监控训练过程
for epoch in tqdm(range(total_epoch), desc="Training Progress"):
    model.fit(x_train_processed_df, y_train)

# 1. 可视化训练过程中损失值变化
plt.plot(model.loss_curve_)
plt.title('Loss Curve during Training')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
# 保存图像
image_name = "Loss Curve during Training"
file_name = f"{current_time}-epochs={total_epoch}-lr={lr}-net={hidden_layer_sizes}-{image_name}.png"
file_path = os.path.join(results_dir, file_name)
plt.savefig(file_path)
plt.show()

# 在测试集上评估模型
y_pred = model.predict(x_test_processed_df)

# 输出均方误差(MSE)和R^2分数
mse = mean_squared_error(y_test, y_pred)  # 均方误差，越低越好
r2 = r2_score(y_test, y_pred)  # 决定系数，越接近于1越好，说明模型拟合的越好
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# 2. 可视化预测结果与真实值的对比
plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # 绘制y=x线
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.grid(True)
# 保存图像
image_name = "True Values vs Predictions"
file_name = f"{current_time}-epochs={total_epoch}-lr={lr}-net={hidden_layer_sizes}-{image_name}.png"
file_path = os.path.join(results_dir, file_name)
plt.savefig(file_path)
plt.show()

# 分析不同特征对预测结果的影响
import shap
# 基于SHAP来分析
explainer = shap.Explainer(model.predict, x_train_processed_df)
shap_values = explainer(x_test_processed_df)
# 绘制并保存SHAP值图
plt.figure()
shap.summary_plot(shap_values, x_test_processed_df, plot_type="bar", show=False)  # 设置show=False以便后续保存
plt.title('SHAP Summary Plot')
# 保存图像
file_path = os.path.join(results_dir, 'shap_summary_plot.png')
plt.savefig(file_path, bbox_inches='tight')  # 使用bbox_inches='tight'以确保图像不被裁剪
plt.close()