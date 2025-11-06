using System;
using MathNet.Numerics.LinearAlgebra;

namespace Optimizer.Core.Common
{
    /// <summary>
    /// Indicates the high-level outcome of a solver run.
    /// </summary>
    public enum SolverResultStatus
    {
        Optimal,
        Infeasible,
        Unbounded,
        IterationLimit,
        TimeLimit,
        Error
    }

    /// <summary>
    /// Encapsulates the result of an optimisation routine.
    /// </summary>
    public sealed class Solution
    {
        public Solution(Vector<double> optimalX, double optimalValue, SolverResultStatus status, int iterations, TimeSpan solveTime)
        {
            OptimalX = optimalX;
            OptimalValue = optimalValue;
            Status = status;
            Iterations = iterations;
            SolveTime = solveTime;
        }

        public Vector<double> OptimalX { get; }

        public double OptimalValue { get; }

        public SolverResultStatus Status { get; }

        public int Iterations { get; }

        public TimeSpan SolveTime { get; }
    }
}
