'''
@Description: 本工具用于估计模型的总参数量和计算FLOPs数
'''
from thop import profile
import torch

# 估计MLP
from MLPs.MLP import MLP
'''
MLP
预估的总参数量为： 402250.0
预估的总FLOPs为： 402048.0
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = MLP().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model1, inputs=(input, ))
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)