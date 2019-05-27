using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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
        public NDArray[] Biases { get; }

        /// <summary>
        /// 神经网络权重
        /// </summary>
        public NDArray[] Weights { get; }

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
    }
}
