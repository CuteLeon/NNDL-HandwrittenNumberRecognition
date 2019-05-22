using System;
using System.Collections.Generic;
using System.Linq;
using NNDL_HandwrittenNumberRecognition.NeuralNetwork.Neurons;
using NNDL_HandwrittenNumberRecognition.Util;
using NumSharp;

namespace NNDL_HandwrittenNumberRecognition.NeuralNetwork
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
        List<List<TNeuron>> NeuronPool = null;

        /// <summary>
        /// 初始化神经网络
        /// </summary>
        /// <param name="neureCounts">神经网络每层神经元个数的集合</param>
        public Network(IEnumerable<int> neureCounts)
        {
            // 初始化神经网络神经元集合
            this.NeuronPool = new List<List<TNeuron>>(
                neureCounts.Select(
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

            Helper.PrintLine("");
            /*
            // 初始化相邻层神经元的权重三维数组
            this.Weights = neureCounts
                .SkipLast(1)
                .Zip(
                    neureCounts
                    .Skip(1))
                .Select(count =>
                    pyRandom.randn(new int[] { count.Second, count.First }).Array
                    .Cast<double>()
                    .Splite(count.First)
                    .Select(values => values.ToArray())
                    .ToArray())
                .ToArray();
             */
        }
    }
}
