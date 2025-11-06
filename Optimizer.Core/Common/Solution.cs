using System;
using MathNet.Numerics.LinearAlgebra;

namespace Optimizer.Core.Common
{
    public enum SolverResultStatus
    {
        Unknown,
        Optimal,
        Infeasible,
        Unbounded,
        IterationLimit,
        Error,
        NotImplemented
    }

    public class Solution
    {
        public Vector<double> OptimalPoint { get; set; }

        public double OptimalValue { get; set; }

        public SolverResultStatus Status { get; set; } = SolverResultStatus.Unknown;

        public int Iterations { get; set; }

        public TimeSpan SolveTime { get; set; }

        public string Message { get; set; }
    }
}
