using System;
using System.Threading;

namespace NNDL.Util
{
    /// <summary>
    /// 通用助手
    /// </summary>
    public static class Helper
    {
        /// <summary>
        /// 输出行
        /// </summary>
        /// <param name="message"></param>
        public static void PrintLine(string message)
            => Console.WriteLine($"{DateTime.Now.ToString("HH:mm:ss.fff")} threadId={Thread.CurrentThread.ManagedThreadId} : {message}");

        /// <summary>
        /// 输出分割线
        /// </summary>
        public static void PrintSplit()
            => Console.WriteLine("——————————————");
    }
}
