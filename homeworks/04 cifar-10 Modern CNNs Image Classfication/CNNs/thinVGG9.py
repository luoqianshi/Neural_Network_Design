import torch
from torch import nn
from d2l import torch as d2l

'''
参考VGG原文将thinVGG11调整为thinVGG9
已经开始收敛在lr=0.01时
但是收敛速度慢，有可能是全连接层的4096这个参数没有修改导致的
需要在做一组把全连接层更轻量化的修改，命名为condenseVGG9
'''

class thinVGG9(nn.Module):
    def __init__(self):
        super(thinVGG9, self).__init__()
        self.features = nn.Sequential(
            # 第一组卷积层
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # (N, 3, 32, 32) → (N, 64, 32, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (N, 64, 32, 32) → (N, 64, 16, 16)

            # 第二组卷积层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (N, 64, 16, 16) → (N, 128, 16, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (N, 128, 16, 16) → (N, 128, 8, 8)

            # 第三组卷积层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (N, 128, 8, 8) → (N, 256, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (N, 256, 8, 8) → (N, 256, 8, 8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (N, 256, 8, 8) → (N, 256, 4, 4)

            # 第四组卷积层
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (N, 256, 4, 4) → (N, 512, 4, 4)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (N, 512, 4, 4) → (N, 512, 4, 4)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 512, 4, 4) → (N, 512, 2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),  # 假设输入图像为 32x32，经过卷积和池化后，特征图大小为 512x2x2
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)  # 输出 10 类
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平
        x = self.classifier(x)
        return x
