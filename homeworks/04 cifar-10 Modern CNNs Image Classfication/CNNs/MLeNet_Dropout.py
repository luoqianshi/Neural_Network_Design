'''
@Author: LQS
@Date: 2024-11-30
@Description:
- 该网络修改自ELeNet目的是为了降低参数规模尽量与MLP保持接近
- 预估的总参数量为： 467718.0
- 预估的总FLOPs为： 822608.0
'''
import torch
import torch.nn as nn

class MLeNet_Dropout(nn.Module):
    def __init__(self):
        super(MLeNet_Dropout, self).__init__()
        # CIFAR-10输入：(3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)  # 修改输出通道数
        # (3, 32, 32) -> (6, 30, 30)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (6, 30, 30) -> (6, 15, 15)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=0)  # 修改输出通道数
        # (6, 15, 15) -> (12, 13, 13)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (12, 13, 13) -> (12, 6, 6)
        self.conv3 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=0)  # 修改输出通道数
        # (12, 6, 6) -> (64, 4, 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (64, 4, 4) -> (64, 2, 2)
        self.fc1 = nn.Linear(in_features=64 * 2 * 2, out_features=128)  # 修改输出神经元数量
        # (64, 2, 2) -> 256 -> 128
        self.dropout1 = nn.Dropout(p=0.5)  # 添加 Dropout 层
        self.fc2 = nn.Linear(in_features=128, out_features=3072)  # 修改输出神经元数量
        # 128 -> 3072
        self.dropout2 = nn.Dropout(p=0.5)  # 添加第二个 Dropout 层
        self.out = nn.Linear(in_features=3072, out_features=10)
        # 3072 -> 10

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x.cuda())))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # 在第一个全连接层后应用 Dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)  # 在第二个全连接层后应用 Dropout
        out = self.out(x)
        return out
