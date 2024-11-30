import torch
import torch.nn as nn

'''
MLP
预估的总参数量为： 402250.0
预估的总FLOPs为： 402048.0
'''

class MLP(nn.Module):  # 2 usages
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(32 * 32 * 3, out_features=128)  # 输入层 -> 第一隐藏层
        self.hidden2 = nn.Linear(in_features=128, out_features=64)  # 第一隐藏层 -> 第二隐藏层
        self.output = nn.Linear(in_features=64, out_features=10)  # 第二隐藏层 -> 输出层

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # 展平输入
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x