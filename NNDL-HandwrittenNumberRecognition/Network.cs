using System.Linq;
using NumSharp;

namespace NNDL_HandwrittenNumberRecognition
{
    /// <summary>
    /// 神经网络
    /// </summary>
    public class Network
    {
        /// <summary>
        /// 神经网络层数
        /// </summary>
        public int NeureLayerCount { get; }

        /// <summary>
        /// 神经网络每层神经元个数
        /// </summary>
        public int[] NeureCounts { get; }

        /// <summary>
        /// 神经网络每层的每个神经元的偏置
        /// </summary>
        public double[][] Biases { get; }

        /// <summary>
        /// 神经网络每层的每个神经元对前一层所有神经元的权重
        /// </summary>
        public double[][][] Weights { get; }

        /// <summary>
        /// 初始化神经网络
        /// </summary>
        /// <param name="neureCounts">神经网络每层神经元个数的集合</param>
        public Network(int[] neureCounts)
        {
            this.NeureLayerCount = neureCounts.Length;
            this.NeureCounts = neureCounts;

            // 高斯(正态)分布 随机数发生器
            var pyRandom = new NumPyRandom();
            // 初始化神经元偏置二维数组
            this.Biases = neureCounts
                .Skip(1)
                .Select(count =>
                    pyRandom.randn(new[] { count, 1 }).Array
                    .Cast<double>()
                    .ToArray())
                .ToArray();
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
        }
    }
}
