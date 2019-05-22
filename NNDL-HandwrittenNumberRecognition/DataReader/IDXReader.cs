using System;
using System.Collections.Generic;
using System.IO;

namespace NNDL_HandwrittenNumberRecognition.DataReader
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
        public IEnumerable<byte[,]> ReadMatrixs(string filePath)
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
            Console.WriteLine($"IDX 文件矩阵数据尺寸：{width} x {height}");

            var blockBuffer = new byte[width, height];
            while (stream.Position < stream.Length)
            {
                for (int heightIndex = 0; heightIndex < height; heightIndex++)
                {
                    byte[] buffer = reader.ReadBytes(width);
                    for (int widthIndex = 0; widthIndex < width; widthIndex++)
                    {
                        blockBuffer[heightIndex, widthIndex] = buffer[widthIndex];
                    }
                }
                yield return blockBuffer;
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
