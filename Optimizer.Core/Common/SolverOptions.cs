using System;

namespace Optimizer.Core.Common
{
    public enum LinearAlgorithm
    {
        Simplex,
        InteriorPoint
    }

    public class SolverOptions
    {
        public double Tolerance { get; set; } = 1e-8;

        public double ConstraintTolerance { get; set; } = 1e-8;

        public int MaxIterations { get; set; } = 1000;

        public TimeSpan? TimeLimit { get; set; }

        public bool Verbose { get; set; }

        public LinearAlgorithm LinearAlgorithm { get; set; } = LinearAlgorithm.Simplex;
    }
}
