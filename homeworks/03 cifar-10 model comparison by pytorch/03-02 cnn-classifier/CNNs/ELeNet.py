'''
@Author: LQS
@Date: 2024-11-30
@Description:
- 该网络迁移自https://blog.csdn.net/qq_40371333/article/details/142586264
- 参数规模：546,602
- FLOPS：10,358,400
'''

import torch
import torch.nn as nn

class ELeNet(nn.Module):
    def __init__(self):
        super(ELeNet, self).__init__()
        # (3, 32, 32) -> (6, 30, 30)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        # (6, 30, 30) -> (6, 15, 15)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (6, 15, 15) -> (16, 13, 13)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        # (16, 13, 13) -> (16, 6, 6)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (16, 6, 6) -> (128, 4, 4)
        self.conv3 = nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=0)
        # (128, 4, 4) -> (128, 2, 2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (128, 2, 2) -> 512 -> 120
        self.fc1 = nn.Linear(in_features=512, out_features=120)
        # 120 -> 4096
        self.fc2 = nn.Linear(in_features=120, out_features=4096)  # out_size=num_filters
        # 4096 -> 10
        self.out = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x.cuda())))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.out(x)
        return out