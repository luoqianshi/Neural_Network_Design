import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据准备：加载MNIST数据集
# 对数据集进行一些处理，数据格式转换、归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 数据集加载并按照batch_size对数据进行批处理
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 2. 模型搭建：定义多层感知机模型
class MLP(nn.Module):  # 2 usages
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(28 * 28, out_features=128)  # 输入层 -> 第一隐藏层
        self.hidden2 = nn.Linear(in_features=128, out_features=64)  # 第一隐藏层 -> 第二隐藏层
        self.output = nn.Linear(in_features=64, out_features=10)  # 第二隐藏层 -> 输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# 初始化模型
model = MLP().to(device)

# 3. 构建损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 记录每个epoch的准确率
accuracy_list = []

# 4. 模型训练及验证（测试）
def train(model, trainloader, criterion, optimizer, epochs=10):  # 2 usages (1 dynamic)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")

        # 每个epoch结束后进行测试并记录准确率
        accuracy = test(model, testloader)
        accuracy_list.append(accuracy)
        
# 测试函数
def test(model, testloader):  # 1 usage
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    return accuracy

# 训练和测试模型
train(model, trainloader, criterion, optimizer, epochs=10)

# 绘制准确率随epoch变化的曲线图
plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, label='accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.grid(True)
plt.show()

# 可视化5张测试样本及其预测
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# 获取前5张图像的预测
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 可视化图像
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for idx in range(5):
    image = images[idx].cpu().squeeze()
    axes[idx].imshow(image, cmap='gray')
    axes[idx].set_title(f"Label: {labels[idx].item()}\Pred: {predicted[idx].item()}")
    axes[idx].axis('off')
plt.show()