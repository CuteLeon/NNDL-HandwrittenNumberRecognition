using System.Collections.Generic;

namespace NNDL.DataReader
{
    /// <summary>
    /// 数据读取接口
    /// </summary>
    public interface IDataReader
    {
        /// <summary>
        /// 读取矩阵
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public IEnumerable<double[,]> ReadMatrixs(string filePath);

        /// <summary>
        /// 扁平读取矩阵
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public IEnumerable<double[]> ReadMatrixsFlattened(string filePath);

        /// <summary>
        /// 读取值
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public IEnumerable<byte> ReadValues(string filePath);
    }
}
