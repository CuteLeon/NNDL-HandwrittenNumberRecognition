using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Accord.IO;

namespace NNDL_HandwrittenNumberRecognition
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
            Console.WriteLine(@"神经网络和深度学习-手写数字识别：
使用三层神经网络结构，28*28=784个输入神经元，15个 隐藏层神经元、10个输出神经元。
输入为：0~1的实数代表每个像素的灰度值，0.0=白色，1.0=黑色。");

            Console.WriteLine("检查数据文件...");
            if (!CheckDataFiles())
            {
                Exit();
            }
            _ = new Network(new[] { 2, 3, 1 });
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

        /// <summary>
        /// 读取IDX文件
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        static IEnumerable<string> ReadIDX(string path)
        {
            IdxReader idxReader = new IdxReader(path);
            yield break;
        }
    }
}
