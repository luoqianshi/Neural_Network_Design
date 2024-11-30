'''
@Author: LQS
@Date: 2024-11-30
@Description:
- 该网络迁移自用于Mnist的LeNet结构
- 预估的总参数量为： 52202.0
- 预估的总FLOPs为： 642000.0
'''

import torch
import torch.nn as nn

# PyTorch默认使用的图像通道顺序是：(batch, channel, height, width)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # (3*32*32)-->(6,28,28)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        # (6,28,28)-->(6,14,14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (6,14,14)-->(16,10,10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # (16,10,10)-->(16,5,5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x.cuda())))
        x = self.pool2(torch.relu(self.conv2(x)))
        # x.size(0) 用于保留batch数不变；
        # -1 自动计算这一维的大小，以确保总元素数量不变
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return out