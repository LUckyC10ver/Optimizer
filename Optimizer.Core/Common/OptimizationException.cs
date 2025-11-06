using System;

namespace Optimizer.Core.Common
{
    [Serializable]
    public class OptimizationException : Exception
    {
        public OptimizationException()
        {
        }

        public OptimizationException(string message)
            : base(message)
        {
        }

        public OptimizationException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
