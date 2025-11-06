using MathNet.Numerics.LinearAlgebra;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Represents the result of evaluating nonlinear constraints.
    /// </summary>
    public sealed class ConstraintEvaluation
    {
        public ConstraintEvaluation(int equalityCount, Vector<double> values)
        {
            EqualityCount = equalityCount;
            Values = values;
        }

        /// <summary>
        /// Number of leading entries in <see cref="Values"/> that correspond to equality constraints.
        /// </summary>
        public int EqualityCount { get; }

        /// <summary>
        /// Constraint values. The first <see cref="EqualityCount"/> entries represent equalities
        /// that should evaluate to zero. Remaining entries represent inequalities that should be
        /// less-or-equal to zero.
        /// </summary>
        public Vector<double> Values { get; }

        /// <summary>
        /// Creates a convenience instance with no constraints.
        /// </summary>
        public static ConstraintEvaluation Empty => new ConstraintEvaluation(0, Vector<double>.Build.Dense(0));
    }
}
