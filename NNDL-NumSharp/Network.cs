using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NNDL.Util;
using NumSharp;

namespace NNDL_NumSharp
{
    /// <summary>
    /// 神经网络
    /// </summary>
    public class Network
    {
        /// <summary>
        /// 神经网络层数
        /// </summary>
        public int LayersCount { get; }

        /// <summary>
        /// 神经网络每一层神经元数量
        /// </summary>
        public int[] NeuronsCounts { get; }

        /// <summary>
        /// 神经网络偏置数组
        /// </summary>
        public NDArray[] Biases { get; protected set; }

        /// <summary>
        /// 神经网络权重
        /// </summary>
        public NDArray[] Weights { get; protected set; }

        /// <summary>
        /// 神经网络
        /// </summary>
        /// <param name="neuronsCounts">神经网络每一层神经元数量</param>
        public Network(int[] neuronsCounts)
        {
            this.LayersCount = neuronsCounts.Length;
            this.NeuronsCounts = neuronsCounts;

            // 高斯(正态)分布 随机数发生器
            var pyRandom = new NumPyRandom();
            this.Biases = neuronsCounts.Skip(1).Select(count => pyRandom.randn(count, 1)).ToArray();
            this.Weights = neuronsCounts.Skip(1).Zip(neuronsCounts.SkipLast(1)).Select(counts => pyRandom.randn(counts.First, counts.Second)).ToArray();
        }

        /// <summary>
        /// 随机梯度下降算法训练
        /// </summary>
        /// <param name="trainDatas">训练数据</param>
        /// <param name="epochs">训练纪元数</param>
        /// <param name="batchSize">批量数据容量</param>
        /// <param name="eta">学习率</param>
        public void StochasticGradientDescent(IEnumerable<(NDArray Image, NDArray Label)> trainDatas, int epochs, int batchSize, double eta)
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
        public void UpdateMiniBatch(IEnumerable<(NDArray Image, NDArray Label)> trainDatas, double eta)
        {
            NDArray[] nablaBiases = this.Biases.Select(array => np.zeros(array.shape)).ToArray();
            NDArray[] nablaWeights = this.Weights.Select(array => np.zeros(array.shape)).ToArray();

            // 遍历小批量训练数据
            foreach (var (Image, Label) in trainDatas)
            {
                // 使用反向传播计算偏置和权重的梯度
                var (deltaNablaBiases, deltaNablaWeight) = this.BackPropogation(Image, Label);

                nablaBiases = nablaBiases.Zip(deltaNablaBiases).Select(b => b.First + b.Second).ToArray();
                nablaWeights = nablaWeights.Zip(deltaNablaWeight).Select(w => w.First + w.Second).ToArray();
            }

            // 使用学习率按比例更新 偏置和权重 到神经网络
            this.Biases = this.Biases.Zip(nablaBiases).Select(b => b.First - (eta / trainDatas.Count() * b.Second)).ToArray();
            this.Weights = this.Weights.Zip(nablaWeights).Select(w => w.First - (eta / trainDatas.Count()) * w.Second).ToArray();
        }

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="image"></param>
        /// <param name="label"></param>
        /// <returns></returns>
        public (NDArray[] deltaNablaBiases, NDArray[] deltaNablaWeight) BackPropogation(NDArray image, NDArray label)
        {
            NDArray[] nablaBiases = this.Biases.Select(array => np.zeros(array.shape)).ToArray();
            NDArray[] nablaWeights = this.Weights.Select(array => np.zeros(array.shape)).ToArray();

            NDArray activation = image;
            NDArray z = null, sp = null;
            List<NDArray> activations = new List<NDArray>() { image };
            List<NDArray> zs = new List<NDArray>();

            foreach (var (bias, weight) in this.Biases.Zip(this.Weights))
            {
                z = np.dot(weight, activation) + bias;
                zs.Add(z);
                activation = this.Sigmoid(z);
                activations.Add(activation);
            }

            NDArray delta = this.CostDerivative(activations[^1], label) * this.SigmoidPrime(zs[^1]);
            nablaBiases[^1] = delta;
            nablaWeights[^1] = np.dot(delta, activations[^2].transpose());

            for (int index = 2; index < this.LayersCount; index++)
            {
                z = zs[^index];
                sp = this.SigmoidPrime(z);
                delta = np.dot(this.Weights[^(index - 1)].transpose(), delta) * sp;
                nablaBiases[^index] = delta;
                nablaWeights[^index] = np.dot(delta, activations[^(index + 1)].transpose());
            }

            return (nablaBiases, nablaWeights);
        }

        /// <summary>
        /// Sigmoid 函数
        /// </summary>
        /// <param name="z"></param>
        /// <returns></returns>
        /// <remarks>NumSharp 暂未实现 np.exp(ndarray)</remarks>
        public NDArray Sigmoid(NDArray z)
            => new NDArray((z.Array as double[]).Select(s => this.Sigmoid(s)).ToArray(), z.shape);

        /// <summary>
        /// Sigmoid 函数
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public double Sigmoid(double source)
            => 1.0 / (1.0 + Math.Exp(-source));

        /// <summary>
        /// Sigmoid 导数
        /// </summary>
        /// <param name="sources"></param>
        /// <returns></returns>
        public NDArray SigmoidPrime(NDArray sources)
            => new NDArray((sources.Array as double[]).Select(s => this.Sigmoid(s) * (1 - this.Sigmoid(s))).ToArray(), sources.shape);

        /// <summary>
        /// 代价函数
        /// </summary>
        /// <param name="actuality">真实值</param>
        /// <param name="expectations">期望值</param>
        /// <returns></returns>
        public NDArray CostDerivative(NDArray actuality, NDArray expectations)
            => actuality - expectations;

        /// <summary>
        /// 使用神经网络前馈输出
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public NDArray FeedForward(NDArray input)
        {
            foreach (var (bias, weight) in this.Biases.Zip(this.Weights))
            {
                input = this.Sigmoid(np.dot(weight, input) + bias);
            }
            return input;
        }
    }
}
