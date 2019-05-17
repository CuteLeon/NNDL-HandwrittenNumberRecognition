using System;

namespace NNDL_HandwrittenNumberRecognition
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine(@"神经网络和深度学习-手写数字识别：
使用三层神经网络结构，28*28=784个输入神经元，15个 隐藏层神经元、10个输出神经元。
输入为：0~1的实数代表每个像素的灰度值，0.0=白色，1.0=黑色。");

            Console.Read();
        }
    }
}
