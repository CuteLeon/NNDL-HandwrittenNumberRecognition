﻿using System.Collections.Generic;

namespace System.Linq
{
    /// <summary>
    /// Linq 扩展
    /// </summary>
    public static class LinqExtension
    {
        /// <summary>
        /// 按步长分拆集合
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="sources"></param>
        /// <param name="stepCount"></param>
        /// <returns></returns>
        public static IEnumerable<IEnumerable<T>> Splite<T>(this IEnumerable<T> sources, int stepCount)
        {
            int index = 0;
            while (index < sources.Count())
            {
                yield return sources.Skip(index).Take(stepCount);
                index += stepCount;
            }
        }
    }
}
