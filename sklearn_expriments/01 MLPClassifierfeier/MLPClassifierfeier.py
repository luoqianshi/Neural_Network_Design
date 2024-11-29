# -- 在该实验中，使用的是sklearn自带的iris数据集，用于分类任务
# -- 鸢尾花数据集一共有150个数据，分为三类，每类50个数据，每个数据包含4个属性：萼片长度、萼片宽度、花瓣长度、花瓣宽度

# 1. 导入必要的库
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 2. 数据集加载并根据数据的2维进行可视化
iris = datasets.load_iris()

print(iris.keys())  # 数据集内容以字典的形式存放，通过key()方法查看数据集包含的属性

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
ax.legend(scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes")
plt.show()

# 3. 划分训练集和测试集
x = iris.data  # 数据集数据内容 [150, 4]
y = iris.target  # 数据集数据标签 [150, ] -- 可通过打印或者调试查看具体内容

# -- 按照7:3随机划分训练集和测试集，random_state为随机种子
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 4. 创建并训练神经网络模型
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, learning_rate_init=0.001, learning_rate="constant",
                    solver="adam", verbose=False)
# -- 隐藏层节点大小为10
# -- 最大迭代次数为1000
# -- 初始学习率为0.001
# -- 学习率更新方式为不变
# -- 优化器选择adam
# -- 训练过程及其默认值，Ctrl+F键直达类定义处可知

# 训练模型
mlp.fit(x_train, y_train)

# -- 可视化训练过程中的损失值变化
plt.plot(mlp.loss_curve_)
plt.title('Loss Curve during Training')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 5. 模型预测
y_pred = mlp.predict(x_test)

# 6. 模型评估
# -- 计算各类别准确率、召回率、F1分数
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -- 计算整体准确率
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))