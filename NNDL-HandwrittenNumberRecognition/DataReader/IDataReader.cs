using System.Collections.Generic;

namespace NNDL_HandwrittenNumberRecognition.DataReader
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
        public IEnumerable<byte[,]> ReadMatrixs(string filePath);

        /// <summary>
        /// 读取值
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public IEnumerable<byte> ReadValues(string filePath);
    }
}
