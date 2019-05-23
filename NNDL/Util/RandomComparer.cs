using System;
using System.Collections.Generic;

namespace NNDL.Util
{
    /// <summary>
    /// 随机排序器
    /// </summary>
    public class RandomComparer : IComparer<object>
    {
        /// <summary>
        /// 懒加载随机数发生器
        /// </summary>
        protected static Lazy<Random> random = new Lazy<Random>(() => new Random(), true);

        /// <summary>
        /// 随机
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public int Compare(object x, object y)
            => random.Value.Next(-1, 1);
    }
}
