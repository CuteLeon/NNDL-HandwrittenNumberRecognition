using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using NNDL.Util;

namespace NNDL_ML.NET
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
            Helper.PrintLine(@"神经网络和深度学习-手写数字识别：");

            Helper.PrintLine("检查数据文件...");
            if (!CheckDataFiles()) Exit();

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
