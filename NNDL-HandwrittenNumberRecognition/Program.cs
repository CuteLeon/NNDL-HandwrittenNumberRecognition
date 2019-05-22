using System;
using System.IO;
using System.Linq;
using NNDL_HandwrittenNumberRecognition.DataReader;

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
            Console.WriteLine(@"神经网络和深度学习-手写数字识别：
使用三层神经网络结构，28*28=784个输入神经元，15个 隐藏层神经元、10个输出神经元。
输入为：0~1的实数代表每个像素的灰度值，0.0=白色，1.0=黑色。");

            Console.WriteLine("检查数据文件...");
            if (!CheckDataFiles())
            {
                Exit();
            }
            Console.WriteLine("创建神经网络...");
            var network = new Network(new[] { 784, 15, 10 });

            Console.WriteLine("读取训练数据...");
            IDataReader reader = new IDXReader();
            foreach (var (imageMatrixs, imageLabel) in reader.ReadMatrixs(TrainImagesPath).Zip(reader.ReadValues(TrainLabelsPath)))
            {
                // 输出图像
                Console.WriteLine($"数字：{imageLabel}");
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
            Console.WriteLine("程序即将退出...");
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
                Console.WriteLine($"缺失文件：\n\t{string.Join("\n\t", filesNotExist)}");
                return false;
            }
        }
    }
}
