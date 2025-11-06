using System;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Configurable parameters for the lightweight SQP-style search used by <see cref="RunSqp"/>.
    /// </summary>
    public sealed class SqpOptions
    {
        /// <summary>
        /// Maximum number of outer iterations performed by the solver.
        /// </summary>
        public int MaxIterations { get; set; } = 200;

        /// <summary>
        /// Target tolerance for both gradient norm and constraint violation.
        /// </summary>
        public double Tolerance { get; set; } = 1e-6;

        /// <summary>
        /// Step length used for the gradient-descent style update.
        /// </summary>
        public double StepSize { get; set; } = 0.1;

        /// <summary>
        /// Penalty factor applied to constraint violations.
        /// </summary>
        public double PenaltyWeight { get; set; } = 50.0;

        /// <summary>
        /// Finite difference step used to approximate gradients of the penalised objective.
        /// </summary>
        public double FiniteDifferenceStep { get; set; } = 1e-5;

        /// <summary>
        /// Optional callback that receives progress updates. The tuple provides
        /// the current iterate, penalised objective, raw objective, and violation magnitude.
        /// </summary>
        public Action<MathNet.Numerics.LinearAlgebra.Vector<double>, double, double, double> ProgressCallback { get; set; }
    }
}
