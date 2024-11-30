'''
@Description: 本工具用于估计模型的总参数量和计算FLOPs数
'''
from thop import profile
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 估计LeNet
from CNNs.LeNet import LeNet
'''
预估的总参数量为： 52202.0
预估的总FLOPs为： 642000.0
'''
model1 = LeNet().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model1, inputs=(input, ))
print('--------------------评估LeNet---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估LeNet---------------------')

# 估计MLeNet
from CNNs.MLeNet import MLeNet
'''
预估的总参数量为： 467718.0
预估的总FLOPs为： 822608.0
'''
model2 = MLeNet().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model2, inputs=(input, ))
print('--------------------评估MLeNet---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估MLeNet---------------------')

# 估计ELeNet
from CNNs.ELeNet import ELeNet
'''
预估的总参数量为： 617754.0
预估的总FLOPs为： 1180648.0
'''
model3 = ELeNet().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model3, inputs=(input, ))
print('--------------------评估ELeNet---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估ELeNet---------------------')