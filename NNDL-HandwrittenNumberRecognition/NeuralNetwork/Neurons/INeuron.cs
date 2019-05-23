namespace NNDL.NeuralNetwork.Neurons
{
    /// <summary>
    /// 神经元接口
    /// </summary>
    public interface INeuron
    {
        /// <summary>
        /// 偏置
        /// </summary>
        public double Bias { get; set; }

        /// <summary>
        /// 前驱输入层权重
        /// </summary>
        public double[] Weight { get; set; }
    }
}
