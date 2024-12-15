import torch
from torch import nn
from d2l import torch as d2l

'''
- 这里是根据原论文复现的VGG-11
'''

class OVGG11(nn.Module):
    def __init__(self):
        super(OVGG11, self).__init__()
        self.features = nn.Sequential(
            # 第一组卷积层
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # (N, 3, 224, 224) → (N, 64, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (N, 64, 224, 224) → (N, 64, 112, 112)

            # 第二组卷积层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (N, 64, 112, 112) → (N, 128, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (N, 128, 112, 112) → (N, 128, 56, 56)

            # 第三组卷积层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (N, 128, 56, 56) → (N, 256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (N, 256, 56, 56) → (N, 256, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (N, 256, 56, 56) → (N, 256, 28, 28)

            # 第四组卷积层
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (N, 256, 28, 28) → (N, 512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (N, 512, 28, 28) → (N, 512, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (N, 512, 28, 28) → (N, 512, 14, 14)

            # 第五组卷积层
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (N, 512, 14, 14) → (N, 512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (N, 512, 14, 14) → (N, 512, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (N, 512, 14, 14) → (N, 512, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 假设输入图像为 224x224，经过卷积和池化后，特征图大小为 512x14x14
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
