# %load network.py
"""
network.py
~~~~~~~~~~
IT WORKS

为前馈神经网络实现的随机梯度下降算法模块。
梯度使用反向传播计算。
代码未被优化仅要求可行。

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        sizes 参数代表神经网络每一层的神经元的数量。
        神经元的偏置和权重使用一个平均值为0、方差为1的高斯分布做初始化。
        神经网络第一层作为输入层，并没有偏置数据。

        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 初始化隐藏层和输出层神经元的偏置，二维数组 [层][神经元]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 初始化隐藏层给输入层的权重和输出层给隐藏层的权重，三维数组 [层][后一层神经元][前一层神经元]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        使用小批量随机梯度下降算法训练神经网络。
        训练数据是一个元组，表示训练的输入和输出。
        如果输入了测试数据，网络将在每个纪元之后输出部分进度。

        Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        # 遍历训练周期 纪元
        for j in range(epochs):
            # 打乱训练数据顺序
            random.shuffle(training_data)
            # 以 mini_batch_size 为步长，将一维训练数据分拆为二维
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            # 遍历小批量数据使用 反向传播 更新网络神经元的偏置和权重
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        通过将使用反向传播的梯度下降应用于单个小批量来更新网络的权重和偏差。
        mini_batch 是一个元组列表。
        eta 是学习率。

        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # 使用神经网络的偏置和权重数据的维度初始化空的偏置和权重向量微分算子
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 遍历小批量训练数据
        for x, y in mini_batch:
            # 使用反向传播计算偏置和权重的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        返回一个元组，表示成本函数 C(x) 的梯度。

        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前馈 feedforward
        activation = x
        activations = [x] # 使用列表逐层储存所有的激活数据 list to store all the activations, layer by layer
        zs = [] # 使用列表逐层储存所有所有的 z 向量 list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 请注意，下面循环中的变量l的用法与本书第2章中的符号稍有不同。
        # 这里，L=1表示神经元的最后一层，L=2表示神经元的最后一层，依此类推。
        # 这是本书中方案的重新编号，在这里使用它是为了利用python可以在列表中使用负索引这一事实。

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        返回神经网络输出正确结果的测试输入数。
        注意，假设神经网络的输出是最后一层中激活率最高的神经元的指数。

        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        返回输出激活的偏导数的向量

        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

#### 其他方法 Miscellaneous functions
def sigmoid(z):
    """ sigmoid 函数 The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """ sigmoid 函数的导数 Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
