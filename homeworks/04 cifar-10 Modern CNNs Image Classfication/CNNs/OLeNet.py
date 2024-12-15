'''
@Author: LQS
@Date: 2024-12-11
@Description:
- 该网络是重新看了李沐老师的d2l后发现自己原本的LeNet其实已经是受AlexNet影响的LeNet了，所以这里进行还原，主要有：
- 1. 使用AvgPool2d而不是MaxPool2d
- 2. 使用Sigmoid激活函数，而不是relu
'''

import torch
import torch.nn as nn

# PyTorch默认使用的图像通道顺序是：(batch, channel, height, width)
# 这里提前用上了MaxPool，原本的LeNet用的是AvgPool，因为输入的尺寸是比较小的，所以求均值能更好利用图片信息。
# 以及原本的LeNet其实用的不是relu函数，而是sigmoid激活函数，所以这里的LeNet其实是受AlexNet影响的LeNet
class OLeNet(nn.Module):
    def __init__(self):
        super(OLeNet, self).__init__()
        # (3*32*32)-->(6,28,28)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        # (6,28,28)-->(6,14,14)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # (6,14,14)-->(16,10,10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # (16,10,10)-->(16,5,5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.sigmoid_(self.conv1(x.cuda())))
        x = self.pool2(torch.sigmoid_(self.conv2(x)))
        # x.size(0) 用于保留batch数不变；
        # -1 自动计算这一维的大小，以确保总元素数量不变
        x = x.reshape(x.size(0), -1)
        x = torch.sigmoid_(self.fc1(x))
        out = self.fc2(x)
        return out
