'''
@Description: 本工具用于估计模型的总参数量和计算FLOPs数
'''
from thop import profile
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 估计LeNet
from CNNs.LeNet import LeNet
'''
预估的总参数量为： 52’202.0
预估的总FLOPs为： 642'000.0
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
预估的总参数量为： 467'718.0
预估的总FLOPs为： 822'608.0
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
预估的总参数量为： 617'754.0
预估的总FLOPs为： 1'180'648.0
'''
model3 = ELeNet().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model3, inputs=(input, ))
print('--------------------评估ELeNet---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估ELeNet---------------------')


# 估计AlexNet
from CNNs.AlexNet import AlexNet
'''
预估的总参数量为： 46'787'978.0
预估的总FLOPs为： 1'005'890'688.0
'''
model4 = AlexNet().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 224, 224).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model4, inputs=(input, ))
print('--------------------评估AlexNet---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估AlexNet---------------------')


# 估计OLeNet
from CNNs.OLeNet import OLeNet
'''
预估的总参数量为： 52'202.0
预估的总FLOPs为： 643'576.0
'''
model5 = OLeNet().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model5, inputs=(input, ))
print('--------------------评估OLeNet---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估OLeNet---------------------')

# 估计thinAlexNet
from CNNs.thinAlexNet import thinAlexNet
'''
AlexNet预估的总参数量为： 46'787'978.0
AlexNet预估的总FLOPs为： 1'005'890'688.0

thinAlexNet预估的总参数量为： 7‘139’722.0
thinAlexNet预估的总FLOPs为： 225‘208’448.0
'''
model6 = thinAlexNet().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model6, inputs=(input, ))
print('--------------------评估thinAlexNet---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估thinAlexNet---------------------')

# 估计VGG11
from CNNs.VGG11 import VGG11
'''
AlexNet预估的总参数量为： 46'787'978.0
AlexNet预估的总FLOPs为： 1'005'890'688.0

VGG11预估的总参数量为： 432'553'546.0
VGG11预估的总FLOPs为： 10'687'848'448.0
'''
model7 = VGG11().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 224, 224).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model7, inputs=(input, ))
print('--------------------评估VGG11---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估VGG11---------------------')

# 估计thinVGG11
from CNNs.thinVGG11 import thinVGG11
'''
AlexNet预估的总参数量为： 46'787'978.0
AlexNet预估的总FLOPs为： 1'005'890'688.0

VGG11预估的总参数量为： 432'553'546.0
VGG11预估的总FLOPs为： 10'687'848'448.0

thinVGG11预估的总参数量为： 29‘900’362.0
thinVGG11预估的总FLOPs为： 234‘594’304.0
'''
model8 = thinVGG11().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model8, inputs=(input, ))
print('--------------------评估thinVGG11---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估thinVGG11---------------------')

# 估计ResNet32
from CNNs.ResNet import ResNet32
'''
AlexNet预估的总参数量为： 46'787'978.0
AlexNet预估的总FLOPs为： 1'005'890'688.0

VGG11预估的总参数量为： 432'553'546.0
VGG11预估的总FLOPs为： 10'687'848'448.0

thinVGG11预估的总参数量为： 29‘900’362.0
thinVGG11预估的总FLOPs为： 234‘594’304.0

ResNet32预估的总参数量为： 466'906.0
ResNet32预估的总FLOPs为： 70'390'464.0
'''
model9 = ResNet32().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model9, inputs=(input, ))
print('--------------------评估ResNet32---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估ResNet32---------------------')

# 估计thinVGG9
from CNNs.thinVGG9 import thinVGG9
'''
AlexNet预估的总参数量为： 46'787'978.0
AlexNet预估的总FLOPs为： 1'005'890'688.0

VGG11预估的总参数量为： 432'553'546.0
VGG11预估的总FLOPs为： 10'687'848'448.0

thinVGG11预估的总参数量为： 29‘900’362.0
thinVGG11预估的总FLOPs为： 234‘594’304.0

--------------------评估thinVGG9---------------------
预估的总参数量为： 29'715'850.0
预估的总FLOPs为： 159'096'832.0
--------------------评估thinVGG9---------------------
'''
model10 = thinVGG9().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model10, inputs=(input, ))
print('--------------------评估thinVGG9---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估thinVGG9---------------------')

# 估计condenseVGG9
from CNNs.condenseVGG9 import condenseVGG9
'''
AlexNet预估的总参数量为： 46'787'978.0
AlexNet预估的总FLOPs为： 1'005'890'688.0

VGG11预估的总参数量为： 432'553'546.0
VGG11预估的总FLOPs为： 10'687'848'448.0

thinVGG11预估的总参数量为： 29‘900’362.0
thinVGG11预估的总FLOPs为： 234‘594’304.0

--------------------评估thinVGG9---------------------
预估的总参数量为： 29'715'850.0
预估的总FLOPs为： 159'096'832.0
--------------------评估thinVGG9---------------------

--------------------评估condenseVGG9---------------------
预估的总参数量为： 7'658'890.0
预估的总FLOPs为： 137'046'016.0
--------------------评估condenseVGG9---------------------
'''
model11 = condenseVGG9().to(device)
# 模拟Cifar-10 数据集的输入效果
input = torch.randn(1, 3, 32, 32).to(device)  # 确保输入张量在同一设备上
macs, params = profile(model11, inputs=(input, ))
print('--------------------评估condenseVGG9---------------------')
print('预估的总参数量为：', params)
print('预估的总FLOPs为：', macs)
print('--------------------评估condenseVGG9---------------------')