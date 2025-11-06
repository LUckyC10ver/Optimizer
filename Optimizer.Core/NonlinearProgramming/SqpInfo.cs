namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Status information returned by <see cref="RunSqp.runSqp"/>.
    /// </summary>
    public sealed class SqpInfo
    {
        /// <summary>
        /// Final objective value (without penalties) reported by the solver.
        /// </summary>
        public double ObjectiveValue { get; set; }

        /// <summary>
        /// Number of outer iterations performed.
        /// </summary>
        public int IterationCount { get; set; }

        /// <summary>
        /// Total number of penalised objective evaluations.
        /// </summary>
        public int FunctionEvaluations { get; set; }

        /// <summary>
        /// Estimated norm of the penalised gradient at the final iterate.
        /// </summary>
        public double GradientNorm { get; set; }

        /// <summary>
        /// Maximum absolute constraint violation measured at the final iterate.
        /// </summary>
        public double ConstraintViolation { get; set; }

        /// <summary>
        /// Textual description of the stop reason.
        /// </summary>
        public string Status { get; set; } = string.Empty;
    }
}
