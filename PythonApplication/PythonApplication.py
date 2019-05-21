import mnist_loader
import network

# 元组形式读取训练数据、验证数据、测试数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# 将训练数据转换为 元组的列表
training_data = list(training_data)

# 创建神经网络
net = network.Network([784, 30, 10])
# 使用 随机梯度下降算法 训练神经网络 (训练数据，训练纪元，小批量数据容量，学习率，测试数据)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
print("success");