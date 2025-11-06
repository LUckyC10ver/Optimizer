using MathNet.Numerics.LinearAlgebra;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Represents the evaluation of nonlinear constraints at a given point.
    /// </summary>
    public sealed class ConstraintEvaluation
    {
        public ConstraintEvaluation(Vector<double> values, int equalityCount, Matrix<double> jacobian = null)
        {
            Values = values;
            EqualityCount = equalityCount;
            Jacobian = jacobian;
        }

        public Vector<double> Values { get; }

        public int EqualityCount { get; }

        public Matrix<double> Jacobian { get; }

        public static ConstraintEvaluation Empty { get; } = new ConstraintEvaluation(null, 0, null);
    }
}
