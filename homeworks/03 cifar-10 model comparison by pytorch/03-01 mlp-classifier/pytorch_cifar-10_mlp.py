import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------超参数定义--------- #
total_epoch = 5  # 训练的总世代数
learning_rate = 0.01  # 学习率
batch_size = 64  # 批处理大小
# ---------超参数定义--------- #


# 1. 数据准备：加载CIFAR-10数据集
# 对数据集进行一些处理，数据格式转换、归一化
# 图像尺寸为 32x32，且有 3 个通道（RGB）
transform = transforms.Compose([
    transforms.ToTensor(),
    # CIFAR-10 的图像是彩色的，因此在归一化时需要为每个通道指定均值和标准差。
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 2. 模型搭建：定义多层感知机模型
from MLPs.MLP import MLP

# 初始化模型
model = MLP().to(device)

# 3. 构建损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 记录每个epoch的损失值
loss_list = []
# 记录每个epoch的准确率
accuracy_list = []
# 初始化数据存储结构
training_results = []

# 4. 模型训练及验证（测试）
def train(model, trainloader, criterion, optimizer, epochs=total_epoch):  # 2 usages (1 dynamic)
    model.train()
    start_time = time.time()  # 记录开始时间
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # progress_bar.set_postfix(loss=running_loss / len(trainloader))

        epoch_loss = running_loss / len(trainloader)
        loss_list.append(epoch_loss)  # 记录每个epoch的损失
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # 每个epoch结束后进行测试并记录准确率（训练集）
        accuracy = validate(model, trainloader)
        accuracy_list.append(accuracy)

        # 记录每个epoch的损失和准确率
        training_results.append((epoch + 1, epoch_loss, accuracy))
    end_time = time.time()  # 记录结束时间
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

# 验证函数（仅用于在模型训练的时候对训练集进行训练集的准确率测试）
def validate(model, dataloader):  # 1 usage
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 测试函数
def test(model, testloader):  # 1 usage
    model.eval()
    start_time = time.time()  # 记录开始时间
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    end_time = time.time()  # 记录结束时间
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"Total Inference Time: {end_time - start_time:.2f} seconds")
    return accuracy, cm


if __name__ == '__main__':
    # 1. 数据集加载并按照batch_size对数据进行批处理
    train_set = torchvision.datasets.CIFAR10(root='.././data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='.././data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 2. 训练和测试模型，并保存模型

    # 创建模型保存目录
    model_dir = './model'
    os.makedirs(model_dir, exist_ok=True)
    
    # 训练和测试模型
    train(model, train_loader, criterion, optimizer, epochs=total_epoch)
    
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # 创建结果目录
    results_dir = f"./results/{current_time}"
    os.makedirs(results_dir, exist_ok=True)
    # 保存模型
    model_path = os.path.join(model_dir, f"model_{current_time}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 计算并打印测试集的准确率和混淆矩阵
    test_accuracy, cm = test(model, test_loader)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    # 显示混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_set.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    # 保存混淆矩阵图像
    plt.savefig(os.path.join(f'./results/{current_time}', f'confusion_matrix_{current_time}.png'))
    plt.show()
    
    # 3. 绘制训练集准确率随epoch变化的曲线图
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy over Epochs')
    plt.grid(True)
    plt.savefig(os.path.join(f'./results/{current_time}', f'training_accuracy_{current_time}.png'))
    plt.show()

    # 4. 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(f'./results/{current_time}', f'training_loss_{current_time}.png'))
    plt.show()

    # 5. 保存训练结果(训练损失和训练准确率)到CSV文件
    df = pd.DataFrame(training_results, columns=["Epoch", "Loss", "Accuracy"])
    csv_file_path = os.path.join(f'./results/{current_time}', f"training_results_{current_time}.csv")
    df.to_csv(csv_file_path, index=False)
    print(f"Training results saved to {csv_file_path}")