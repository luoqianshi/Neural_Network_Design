import torch
from torch import nn
from d2l import torch as d2l
'''
* 尝试专门为32x32尺寸的输入构造一个thinAlexNet
* 这个是我自己修改的网络结构，效果如何，其实我自己也不清楚，
* 大概就是模仿着AlexNet的结构，结合LeNet的思想来为小尺寸图片输入作适配，并且进行了一定的参数轻量化
'''

class thinAlexNet(nn.Module):
    def __init__(self):
        super(thinAlexNet, self).__init__()
        self.net = nn.Sequential(
            # 由于输入的尺寸为32x32所以不需要太大的卷积核尺寸，这里采用5x5;
            # 并且采用步长为1，避免快速地把尺寸放缩过小;
            # 输出通道的数目远大于LeNet
            # (3*32*32)——>(96*30*30)
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=1), nn.ReLU(),
            # (96*30*30)——>(96*15*15)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 使用填充为1来使得输入与输出的高和宽一致，且增大输出通道数
            # (96*15*15)——>(256*13*13)
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=1), nn.ReLU(),
            # (256*13*13)——>(256*6*6)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层【不用于减少输入的高度和宽度】
            # (256*6*6)——>(384*6*6)
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            # (384*6*6)——>(384*6*6)
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            # (384*6*6)——>(256*6*6)
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            # (256*6*6)——>(256*3*3)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            # 这里，全连接层的输出数量更改为1024，是原本AlexNet的1/4。使用dropout层来减轻过拟合
            nn.Linear(256 * 3 * 3, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层。由于这里使用Cifar-10，所以用类别数为10
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        return self.net(x)