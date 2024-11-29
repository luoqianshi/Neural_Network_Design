# 使用torchvision来加载mnist数据集
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from tqdm import tqdm
from datetime import datetime

# -------这里是超参数和全局变量定义------- #
# 训练的总世代数
total_epoch = 50
# 初始学习率
lr = 0.0001
# batch_size 默认值为 'auto'，即 min(200, n_samples)
batch_size = 'auto'
# MLP模型隐藏层的网络形状
hidden_layer_sizes = (25, 15, 5)
# 激活函数
activation = 'relu'
# 优化器（求解器）
solver = 'adam'
# 模型权重参数暖启动
# 当 warm_start=True 时，模型在每次调用 fit 方法时不会重新初始化参数，
# 而是继续使用上一次训练的参数。这意味着你可以在多次调用 fit 方法时累积训练效果。
warm_start=True
# 初始化损失列表
train_loss_values = []
val_loss_values = []
# 初始化精度列表
train_accuracy_values = []
val_accuracy_values = []
# -------这里是超参数和全局变量定义------- #

# -------实验结果保存相关------- #
# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
# 创建结果目录
results_dir = f"./results/{current_time}"
os.makedirs(results_dir, exist_ok=True)
# -------实验结果保存相关------- #


# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    # 这个转换将图像从 PIL 图像或 NumPy 数组转换为 PyTorch 张量。
    # 它会将图像的像素值从 [0, 255] 范围缩放到 [0.0, 1.0] 范围。
    transforms.Normalize((0.5,), (0.5,))  # 归一化
    # 接收两个参数：均值和标准差。这里的 (0.5,) 表示对每个通道减去均值 0.5，然后除以标准差 0.5。
    # 结果是将像素值从 [0.0, 1.0] 范围缩放到 [-1.0, 1.0] 范围。
])

# 加载MNIST数据集
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 打印原始数据集的大小
print(f"Original training dataset size: {len(mnist_train)}")
print(f"Original test dataset size: {len(mnist_test)}")

# 将数据转换为sklearn格式并展平
X_train = mnist_train.data.view(-1, 28*28).numpy()
y_train = mnist_train.targets.numpy()

X_test = mnist_test.data.view(-1, 28*28).numpy()
y_test = mnist_test.targets.numpy()

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 打印划分后的数据集大小
print(f"Training dataset size after split: {len(X_train)}")
print(f"Validation dataset size: {len(X_val)}")
print(f"Test dataset size: {len(X_test)}")

# 可视化一些MNIST图像
def visualize_data(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    for i in range(num_images):
        image, label = dataset[i]
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

# 显示训练集中的前5个图像
# visualize_data(mnist_train)


# 使用tqdm显示训练进度
model = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver=solver,
    max_iter=1,
    random_state=42,
    warm_start=warm_start,
    learning_rate_init=lr,
    batch_size=batch_size
)

# 训练模型并记录损失
for epoch in tqdm(range(total_epoch), desc="Training Progress"):
    model.fit(X_train, y_train)

    # 计算训练集的损失
    y_train_pred_proba = model.predict_proba(X_train)
    train_loss = log_loss(y_train, y_train_pred_proba)
    train_loss_values.append(train_loss)
    # 计算训练集的精度
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    train_accuracy_values.append(train_accuracy)

    # 计算验证集的损失
    y_val_pred_proba = model.predict_proba(X_val)
    val_loss = log_loss(y_val, y_val_pred_proba)
    val_loss_values.append(val_loss)
    # 计算验证集的精度
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    val_accuracy_values.append(val_accuracy)

    # 更新tqdm描述信息
    tqdm.write(f"\nEpoch {epoch+1}/{total_epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
               f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# 1. 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(0, total_epoch), train_loss_values, marker='o', linestyle='-', label='Training Loss')
plt.plot(range(0, total_epoch), val_loss_values, marker='x', linestyle='--', label='Validation Loss')
plt.title('Training and Validation Loss Convergence')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
# 保存图像
image_name = "Training and Validation Loss Convergence"
file_name = f"{current_time}-epochs={total_epoch}-lr={lr}-net={hidden_layer_sizes}-{image_name}.png"
file_path = os.path.join(results_dir, file_name)
plt.savefig(file_path)
plt.show()

# 2. 绘制精度曲线
plt.figure(figsize=(10, 5))
plt.plot(range(0, total_epoch), train_accuracy_values, marker='o', linestyle='-', label='Training Accuracy')
plt.plot(range(0, total_epoch), val_accuracy_values, marker='x', linestyle='--', label='Validation Accuracy')
plt.title('Training and Validation Accuracy Convergence')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# 保存图像
image_name = "Training and Validation Accuracy Convergence"
file_name = f"{current_time}-epochs={total_epoch}-lr={lr}-net={hidden_layer_sizes}-{image_name}.png"
file_path = os.path.join(results_dir, file_name)
plt.savefig(file_path)
plt.show()

# 预测测试集
y_pred = model.predict(X_test)

# 输出准确率、分类报告和混淆矩阵
print("Accuracy:", accuracy_score(y_test, y_pred))

# 3. 输出分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))
report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()

# 绘制热力图来展示分类报告
plt.figure(figsize=(10, 6))
# sns.heatmap(df.iloc[:-3, :-1], annot=True, cmap='Blues', fmt='.2f')
sns.heatmap(df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.2f')  # 保留所有的行
plt.title('Classification Report')
# 保存图像
image_name = "Classification Report"
file_name = f"{current_time}-epochs={total_epoch}-lr={lr}-net={hidden_layer_sizes}-{image_name}.png"
file_path = os.path.join(results_dir, file_name)
plt.savefig(file_path)
plt.show()

# 4. 输出混淆矩阵
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 假设 y_true 和 y_pred 是你的真实标签和预测标签
cm = confusion_matrix(y_test, y_pred)
x_class_names = np.unique(y_pred)
y_class_names = np.unique(y_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=x_class_names, yticklabels=y_class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
# 保存图像
image_name = "Confusion Matrix"
file_name = f"{current_time}-epochs={total_epoch}-lr={lr}-net={hidden_layer_sizes}-{image_name}.png"
file_path = os.path.join(results_dir, file_name)
plt.savefig(file_path)
plt.show()