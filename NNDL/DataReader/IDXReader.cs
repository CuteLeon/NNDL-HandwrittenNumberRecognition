using System.Collections.Generic;
using System.IO;
using System.Linq;
using NNDL.Util;

namespace NNDL.DataReader
{
    /// <summary>
    /// IDX 读取器
    /// </summary>
    public class IDXReader : IDataReader
    {
        /// <summary>
        /// 读取矩阵
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public IEnumerable<double[,]> ReadMatrixs(string filePath)
        {
            // IDX 文件前16个字节：前八个字节为魔法数字，后八个字节分别为数据矩阵尺寸
            using FileStream stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using BinaryReader reader = new BinaryReader(stream);

            _ = stream.Seek(8, SeekOrigin.Begin);
            int width = 0;
            for (int index = 0; index < 4; index++)
            {
                width = (width << 8) + reader.ReadByte();
            }
            int height = 0;
            for (int index = 0; index < 4; index++)
            {
                height = (height << 8) + reader.ReadByte();
            }
            Helper.PrintLine($"IDX 文件矩阵数据尺寸：{width} x {height}");

            var blockBuffer = new double[width, height];
            var buffer = new byte[width];
            while (stream.Position < stream.Length)
            {
                for (int heightIndex = 0; heightIndex < height; heightIndex++)
                {
                    buffer = reader.ReadBytes(width);
                    for (int widthIndex = 0; widthIndex < width; widthIndex++)
                    {
                        blockBuffer[heightIndex, widthIndex] = buffer[widthIndex] / (double)255;
                    }
                }
                yield return blockBuffer;
            }
        }

        /// <summary>
        /// 扁平读取矩阵
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public IEnumerable<double[]> ReadMatrixsFlattened(string filePath)
        {
            // IDX 文件前16个字节：前八个字节为魔法数字，后八个字节分别为数据矩阵尺寸
            using FileStream stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using BinaryReader reader = new BinaryReader(stream);

            _ = stream.Seek(8, SeekOrigin.Begin);
            int width = 0;
            for (int index = 0; index < 4; index++)
            {
                width = (width << 8) + reader.ReadByte();
            }
            int height = 0;
            for (int index = 0; index < 4; index++)
            {
                height = (height << 8) + reader.ReadByte();
            }
            Helper.PrintLine($"IDX 文件矩阵数据尺寸：{width} x {height}");

            int blockLength = width * height;
            var buffer = new byte[blockLength];
            var result = new double[blockLength];
            while (stream.Position < stream.Length)
            {
                buffer = reader.ReadBytes(blockLength);
                result = buffer.Select(v => v / (double)255).ToArray();
                yield return result;
            }
        }

        /// <summary>
        /// 读取值
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public IEnumerable<byte> ReadValues(string filePath)
        {
            // IDX 文件前八个字节为魔法数字
            using FileStream stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using BinaryReader reader = new BinaryReader(stream);

            _ = stream.Seek(8, SeekOrigin.Begin);
            while (stream.Position < stream.Length)
            {
                yield return reader.ReadByte();
            }
        }
    }
}
