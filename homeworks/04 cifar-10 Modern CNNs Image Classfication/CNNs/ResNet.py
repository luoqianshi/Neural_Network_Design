import torch
from torch import nn

'''
- 其实在讨论ResNet之前，可以考虑再去研究研究NIN或者GoogleInception之类的网络，这样才能更好地理解其中的细节
- 以及BN层的效果其实是需要额外去验证的，比如可以考虑在VGG网络上进行验证
- 有空的时候多去搜搜别人的代码是怎么写的，有很多有价值的技术细节要学习下来（找到一种比较好的学习渠道或者环境）
'''

class BasicBlock(nn.Module):
    """A basic block for ResNet, consisting of two convolutional layers with a skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        # 当跨层，特征通道出现变动的时候，需要基于1x1的卷积来进行特征通道数的变化，以此实现跳接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # Add the shortcut
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet model with 32 layers for CIFAR-10."""
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 16
        # Input: (3, 32, 32) -> Output: (16, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  # Output: (16, 32, 32)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  # Output: (32, 16, 16)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # Output: (64, 8, 8)
        # 自适应平均池化的特点是可以根据输入的特征图大小自动调整池化的窗口大小，以输出指定的形状。在这里，目标输出形状是(1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (64, 1, 1)
        self.fc = nn.Linear(64, 10)  # Output: (10)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet32():
    return ResNet(BasicBlock, [5, 5, 5])  # 5 blocks for each layer
