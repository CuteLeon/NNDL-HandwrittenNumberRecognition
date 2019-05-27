using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NNDL.DataReader;
using NNDL.Util;

namespace NNDL_NumSharp
{
    class Program
    {
        static readonly string DataDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas");
        static readonly string TrainImagesPath = Path.Combine(DataDirectory, "train-images.idx3-ubyte");
        static readonly string TrainLabelsPath = Path.Combine(DataDirectory, "train-labels.idx1-ubyte");
        static readonly string T10kImagesPath = Path.Combine(DataDirectory, "t10k-images.idx3-ubyte");
        static readonly string T10kLabelsPath = Path.Combine(DataDirectory, "t10k-labels.idx1-ubyte");

        static void Main(string[] args)
        {
            Helper.PrintLine("NNDL-NumSharp");
            Helper.PrintLine("检查数据文件...");
            if (!CheckDataFiles()) Exit(-1);

            Helper.PrintLine("读取训练数据...");
            MNISTReader reader = new MNISTReader();
            var trainImages = reader.ReadMatrixsFlattened(TrainImagesPath).ToArray();
            var trainLabels = reader.ReadValues(TrainLabelsPath).Select(value => { var values = Enumerable.Repeat<double>(0.0, 10).Cast<double>().ToArray(); values[value] = 1; return values; }).ToArray();

            Helper.PrintLine("创建神经网络...");
            var network = new Network(new[] { 784, 30, 10 });

            Stopwatch stopwatch = new Stopwatch();
            Helper.PrintLine("随机梯度下降算法训练神经网络...");
            Helper.PrintSplit();
            stopwatch.Start();

            /*
            network.StochasticGradientDescent(
                trainImages.Zip(trainLabels),
                3,
                10,
                3.0);
             */

            stopwatch.Stop();
            Helper.PrintSplit();
            Helper.PrintLine($"神经网络训练结束，耗时：{stopwatch.Elapsed.ToString()}");

            Exit();
        }

        static void Exit(int code = 0)
        {
            Helper.PrintLine("程序即将退出...");
            Console.Read();
            Environment.Exit(code);
        }

        static bool CheckDataFiles()
        {
            var filesNotExist = new[]
            {
                TrainImagesPath,
                TrainLabelsPath,
                T10kImagesPath,
                T10kLabelsPath,
            }.Where(path => !File.Exists(path))
            .Select(path => Path.GetFileName(path))
            .ToArray();
            if (filesNotExist.Length == 0)
            {
                return true;
            }
            else
            {
                Helper.PrintLine($"缺失文件：\n\t{string.Join("\n\t", filesNotExist)}");
                return false;
            }
        }
    }
}
