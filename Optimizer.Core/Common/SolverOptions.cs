using System;

namespace Optimizer.Core.Common
{
    /// <summary>
    /// Provides a shared set of configuration knobs for the linear, quadratic, and integer solvers.
    /// </summary>
    public class SolverOptions
    {
        /// <summary>
        /// General tolerance used for feasibility and optimality checks.
        /// </summary>
        public double Tolerance { get; set; } = 1e-8;

        /// <summary>
        /// Maximum number of iterations executed by iterative algorithms.
        /// </summary>
        public int MaxIterations { get; set; } = 10_000;

        /// <summary>
        /// Optional time limit that, when reached, stops the algorithm and returns the best known solution.
        /// </summary>
        public TimeSpan? TimeLimit { get; set; }
            = null;

        /// <summary>
        /// When set to <c>true</c> solvers emit additional diagnostic information via the <see cref="DiagnosticsWriter"/>.
        /// </summary>
        public bool Verbose { get; set; }
            = false;

        /// <summary>
        /// Optional sink for textual diagnostics.
        /// </summary>
        public System.IO.TextWriter DiagnosticsWriter { get; set; }
            = null;
    }
}
