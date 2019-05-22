using System;
using System.IO;
using System.Linq;
using NNDL_HandwrittenNumberRecognition.DataReader;
using NNDL_HandwrittenNumberRecognition.NeuralNetwork;
using NNDL_HandwrittenNumberRecognition.NeuralNetwork.Neurons;
using NNDL_HandwrittenNumberRecognition.Util;

namespace NNDL_HandwrittenNumberRecognition
{
    class Program
    {
        static readonly string DataDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas");
        static readonly string TrainImagesPath = Path.Combine(DataDirectory, "train-images.idx3-ubyte");
        static readonly string TrainLabelsPath = Path.Combine(DataDirectory, "train-labels.idx1-ubyte");
        static readonly string T10kImagesPath = Path.Combine(DataDirectory, "t10k-images.idx3-ubyte");
        static readonly string T10kLabelsPath = Path.Combine(DataDirectory, "t10k-labels.idx1-ubyte");

        static void Main()
        {
            Helper.PrintLine(@"神经网络和深度学习-手写数字识别：");

            Helper.PrintLine("检查数据文件...");
            if (!CheckDataFiles()) Exit();

            Helper.PrintLine("创建神经网络...");
            var network = new Network<SigmoidNeuron>(new[] { 784, 15, 10 });

            Helper.PrintLine("读取训练数据...");
            IDataReader reader = new IDXReader();
            foreach (var (imageMatrixs, imageLabel) in reader.ReadMatrixs(TrainImagesPath).Zip(reader.ReadValues(TrainLabelsPath)))
            {
                // 输出图像
                Helper.PrintLine($"数字：{imageLabel}");
                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        Console.Write(imageMatrixs[y, x] < 128 ? " " : "o");
                    }
                    Console.Write('\n');
                }
                Console.Read();
            }

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
