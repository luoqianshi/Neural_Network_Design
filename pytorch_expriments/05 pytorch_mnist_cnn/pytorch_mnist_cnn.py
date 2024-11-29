import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义卷积神经网络模型
class CNN(nn.Module):  # 2 usages
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积：输入1通道，输出32通道，卷积核大小3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积：输入32通道，输出64通道，卷积核大小3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层：输入7*7*64，输出128
        self.fc1 = nn.Linear(7 * 7 * 64, out_features=128)
        # 全连接层：输入128，输出10（10类）
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型，损失函数和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

import os
from PIL import Image

# 定义图片路径
image_folder = './pic'

# 获取文件夹中所有图片文件
images_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]

# 加载和预处理图片的转换方式(transforms中图像处理函数基本上都适用了PIL格式)
preprocess = transforms.Compose([
    transforms.Grayscale(),  # 转换为灰度图
    transforms.Resize((28, 28)),  # 调整图像大小为28x28
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

# 加载自制图像并进行预测
def predict(model):  # 1 usage
    images = []
    predictions = []
    # 对每张图片进行加载、预处理和预测
    model.eval()
    with torch.no_grad():
        for img_path in images_paths:
            # 加载图片
            image = Image.open(img_path)
            # 预处理图片，增加批次维度，移动到设备上 [1,1,28,28]
            image = preprocess(image).unsqueeze(0).to(device)
            # 进行预测
            output = model(image)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
            images.append(image.cpu().squeeze())

    return images, predictions

# 调用函数获取图像和预测结果
images, predictions = predict(model)

# 可视化图像和预测结果
fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
for idx, (image, prediction) in enumerate(zip(images, predictions)):
    axes[idx].imshow(image, cmap='gray')
    axes[idx].set_title(f'Pred: {prediction}')
    axes[idx].axis('off')
plt.show()