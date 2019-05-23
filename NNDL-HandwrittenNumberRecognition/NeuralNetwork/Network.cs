﻿using System;
using System.Collections.Generic;
using System.Linq;
using NNDL_HandwrittenNumberRecognition.NeuralNetwork.Neurons;
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
    }
}
