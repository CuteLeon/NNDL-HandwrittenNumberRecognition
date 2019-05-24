using System;
using System.Collections.Generic;
using System.Linq;
using NNDL.NeuralNetwork.Neurons;
using NNDL.Util;
using NumSharp;

namespace NNDL.NeuralNetwork
{
    /// <summary>
    /// 神经网络
    /// </summary>
    public class Network<TNeuron>
        where TNeuron : INeuron
    {
        /// <summary>
        /// 神经元集合
        /// </summary>
        readonly List<List<TNeuron>> NeuronPool = null;

        /// <summary>
        /// 初始化神经网络
        /// </summary>
        /// <param name="neuronCounts">神经网络每层神经元个数的集合</param>
        public Network(IEnumerable<int> neuronCounts)
        {
            // 初始化神经网络神经元集合
            this.NeuronPool = new List<List<TNeuron>>(
                neuronCounts.Select(
                    count =>
                    Enumerable.Range(0, count).
                    Select(index => Activator.CreateInstance<TNeuron>())
                    .ToList())
                );

            // 高斯(正态)分布 随机数发生器
            var pyRandom = new NumPyRandom();

            // 为非输入层初始化偏置
            _ = this.NeuronPool.Skip(1).All(neurons =>
            {
                foreach (var (neuron, bias) in neurons.Zip(pyRandom.randn(new[] { neurons.Count, 1 }).Array as double[]))
                {
                    neuron.Bias = bias;
                }

                return true;
            });

            // 为非输入层初始化对前一层神经元输入的权重
            _ = this.NeuronPool.SkipLast(1).Zip(this.NeuronPool.Skip(1)).All(neurons =>
            {
                var weights = pyRandom.randn(new int[] { neurons.Second.Count, neurons.First.Count });
                for (int index = 0; index < neurons.Second.Count; index++)
                {
                    neurons.Second[index].Weight = weights[index].Array as double[];
                }

                return true;
            });
        }

        /// <summary>
        /// 随机梯度下降算法训练
        /// </summary>
        /// <param name="trainDatas">训练数据</param>
        /// <param name="epochs">训练纪元数</param>
        /// <param name="batchSize">批量数据容量</param>
        /// <param name="eta">学习率</param>
        public void StochasticGradientDescent(IEnumerable<(double[] Image, double[] Label)> trainDatas, int epochs, int batchSize, double eta)
        {
            Helper.PrintLine($"训练数据数量：{trainDatas.Count()}，训练纪元：{epochs}，Mini批量数量：{batchSize}，学习率：{eta.ToString("N2")}");
            Helper.PrintSplit();

            // 使用随机比较器打乱训练数据顺序
            trainDatas = trainDatas.OrderBy(data => data, new RandomComparer());

            // 遍历训练纪元
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Helper.PrintLine($"<<< 训练第 {epoch} 纪元 >>>");
                foreach (var batchs in trainDatas.Splite(batchSize))
                {
                    // 使用小批量训练数据更新神经网络的权重和偏置
                    this.UpdateMiniBatch(batchs, eta);
                }
            }
        }

        /// <summary>
        /// 使用小批量数据更新网络神经元的偏置和权重
        /// </summary>
        /// <param name="trainDatas">小批量训练数据</param>
        /// <param name="eta">学习率</param>
        public void UpdateMiniBatch(IEnumerable<(double[] Image, double[] Label)> trainDatas, double eta)
        {
            // 定义微分变量数组
            double[][] nablaBiases = this.NeuronPool.Skip(1).Select(neurons => new double[neurons.Count]).ToArray();
            double[][][] nablaWeight = this.NeuronPool.Skip(1).Select(neurons => neurons.Select(neuron => new double[neuron.Weight.Length]).ToArray()).ToArray();

            foreach (var (Image, Label) in trainDatas)
            {
                // 使用反向传播计算神经网络权重和偏置的偏差量
                var (deltaNablaBiases, deltaNablaWeight) = this.BackPropogation(Image, Label);
                for (int index = 0; index < nablaBiases.Length; index++)
                {
                    nablaBiases[index] = nablaBiases[index].Zip(deltaNablaBiases[index]).Select(bias => bias.First + bias.Second).ToArray();
                }
                for (int index1 = 0; index1 < nablaWeight.Length; index1++)
                {
                    for (int index2 = 0; index2 < nablaWeight[index1].Length; index2++)
                    {
                        nablaWeight[index1][index2] = nablaWeight[index1][index2].Zip(deltaNablaWeight[index1][index2]).Select(weight => weight.First + weight.Second).ToArray();
                    }
                }
            }

            for (int index1 = 0; index1 < nablaBiases.Length; index1++)
            {
                for (int index2 = 0; index2 < nablaBiases[index1].Length; index2++)
                {
                    this.NeuronPool[index1 + 1][index2].Bias = this.NeuronPool[index1 + 1][index2].Bias - (eta / trainDatas.Count()) * nablaBiases[index1][index2];
                }
            }

            for (int index1 = 0; index1 < nablaWeight.Length; index1++)
            {
                for (int index2 = 0; index2 < nablaWeight[index1].Length; index2++)
                {
                    for (int index3 = 0; index3 < nablaWeight[index1][index2].Length; index3++)
                    {
                        this.NeuronPool[index1 + 1][index2].Weight[index3] = this.NeuronPool[index1 + 1][index2].Weight[index3] - (eta / trainDatas.Count()) * nablaWeight[index1][index2][index3];
                    }
                }
            }
        }

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="image"></param>
        /// <param name="label"></param>
        /// <returns></returns>
        public (double[][] deltaNablaBiases, double[][][] deltaNablaWeight) BackPropogation(double[] image, double[] label)
        {
            // 定义微分变量数组
            double[][] deltaNablaBiases = this.NeuronPool.Skip(1).Select(neurons => new double[neurons.Count]).ToArray();
            double[][][] deltaNablaWeight = this.NeuronPool.Skip(1).Select(neurons => neurons.Select(neuron => new double[neuron.Weight.Length]).ToArray()).ToArray();

            double[] z = null;
            double[] activation = image;
            List<double[]> activations = new List<double[]>() { image };
            List<double[]> zs = new List<double[]>();

            // 遍历网络非输入层
            foreach (var neurons in this.NeuronPool.Skip(1))
            {
                // 计算带权和偏置的输入
                z = (np.dot(
                    new NDArray(neurons.Select(n => n.Weight).Join().ToArray(), new Shape(new int[] { neurons.Count, neurons.First().Weight.Length })),
                    new NDArray(activation, new Shape(new int[] { activation.Length, 1 })))
                    + new NDArray(neurons.Select(n => n.Bias).ToArray(), new Shape(new int[] { neurons.Count, 1 }))).Array as double[];
                // 保存带权输入
                zs.Add(z);
                // 计算当前层神经元输出
                activation = this.Sigmoid(z);
                // 记录下一层的输入
                activations.Add(activation);
            }

            // 使用代价函数和Sigmoid导数函数计算输出偏差
            double[] deltaOutput = this.CostDerivative(activations[^1], label)
                .Zip(this.SigmoidPrime(zs[^1]))
                .Select(value => value.First * value.Second)
                .ToArray();

            // 记录输出层偏置偏差
            deltaNablaBiases[^1] = deltaOutput;
            // 记录输出层权重偏差
            var deltaWeight = np.dot(
                new NDArray(deltaOutput, new Shape(new int[] { deltaOutput.Length, 1 })),
                // 整理矩阵需要 [转置]
                new NDArray(activations[^2], new Shape(new int[] { 1, activations[^2].Length })));
            for (int index = 0; index < deltaWeight.shape[0]; index++)
            {
                deltaNablaWeight[^1][index] = deltaWeight[index].Array as double[];
            }

            foreach (int index in Enumerable.Range(2, this.NeuronPool.Count - 2))
            {
                z = zs[^index];
                var sp = this.SigmoidPrime(z);

                deltaOutput = (np.dot(
                    // 这里需要转置
                    new NDArray(this.NeuronPool[^(index - 1)].Select(n => n.Weight).Join().ToArray(), new Shape(new int[] { this.NeuronPool[^(index - 1)].First().Weight.Length, this.NeuronPool[^(index - 1)].Count })),
                    new NDArray(deltaOutput, new Shape(new int[] { deltaOutput.Length, 1 })))
                    * new NDArray(sp, new Shape(sp.Length, 1)))
                    .Array as double[];

                deltaNablaBiases[^index] = deltaOutput;
                deltaWeight = np.dot(
                   new NDArray(deltaOutput, new Shape(new int[] { deltaOutput.Length, 1 })),
                   // 整理矩阵需要 [转置]
                   new NDArray(activations[^(index + 1)], new Shape(new int[] { 1, activations[^(index + 1)].Length })));
                for (int i = 0; i < deltaWeight.shape[0]; i++)
                {
                    deltaNablaWeight[^index][i] = deltaWeight[i].Array as double[];
                }
            }

            return (deltaNablaBiases, deltaNablaWeight);
        }

        /// <summary>
        /// Sigmoid 函数
        /// </summary>
        /// <param name="sources"></param>
        /// <returns></returns>
        public double[] Sigmoid(IEnumerable<double> sources)
            => sources.Select(s => this.Sigmoid(s)).ToArray();

        /// <summary>
        /// Sigmoid 函数
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public double Sigmoid(double source)
            => 1.0 / (1.0 + Math.Exp(-source));

        /// <summary>
        /// 代价函数
        /// </summary>
        /// <param name="actuality">真实值</param>
        /// <param name="expectations">期望值</param>
        /// <returns></returns>
        public double[] CostDerivative(IEnumerable<double> actuality, IEnumerable<double> expectations)
            => actuality.Zip(expectations).Select(tuple => tuple.First - tuple.Second).ToArray();

        /// <summary>
        /// Sigmoid 导数
        /// </summary>
        /// <param name="sources"></param>
        /// <returns></returns>
        public double[] SigmoidPrime(IEnumerable<double> sources)
        {
            return sources.Select(value => this.Sigmoid(value) * (1 - this.Sigmoid(value))).ToArray();
        }
    }
}
