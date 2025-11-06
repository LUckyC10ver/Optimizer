using System;

namespace Optimizer.Core.Common
{
    /// <summary>
    /// Base exception type for all optimisation errors raised by the library.
    /// </summary>
    public class OptimizationException : Exception
    {
        public OptimizationException()
        {
        }

        public OptimizationException(string message) : base(message)
        {
        }

        public OptimizationException(string message, Exception innerException) : base(message, innerException)
        {
        }
    }
}
