using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.QuadraticProgramming
{
    /// <summary>
    /// Encapsulates a dense quadratic programming model of the form
    /// 1/2 x'Qx + c'x subject to optional linear constraints.
    /// </summary>
    public sealed class QuadraticProblem
    {
        public QuadraticProblem(
            Matrix<double> q,
            Vector<double> c,
            Matrix<double> inequalityMatrix = null,
            Vector<double> inequalityVector = null,
            bool isMinimisation = true,
            Matrix<double> equalityMatrix = null,
            Vector<double> equalityVector = null,
            Vector<double> lowerBounds = null,
            Vector<double> upperBounds = null,
            Vector<double> initialGuess = null)
        {
            Q = q ?? throw new OptimizationException("Quadratic matrix Q cannot be null.");
            C = c ?? throw new OptimizationException("Linear vector c cannot be null.");
            InequalityMatrix = inequalityMatrix;
            InequalityVector = inequalityVector;
            EqualityMatrix = equalityMatrix;
            EqualityVector = equalityVector;
            LowerBounds = lowerBounds;
            UpperBounds = upperBounds;
            InitialGuess = initialGuess;
            IsMinimisation = isMinimisation;
        }

        public Matrix<double> Q { get; }

        public Vector<double> C { get; }

        public Matrix<double> InequalityMatrix { get; }

        public Vector<double> InequalityVector { get; }

        public Matrix<double> EqualityMatrix { get; }

        public Vector<double> EqualityVector { get; }

        public Vector<double> LowerBounds { get; }

        public Vector<double> UpperBounds { get; }

        public Vector<double> InitialGuess { get; }

        public bool IsMinimisation { get; }

        public void Validate()
        {
            if (!Q.RowCount.Equals(Q.ColumnCount))
            {
                throw new OptimizationException("Matrix Q must be square.");
            }

            if (Q.ColumnCount != C.Count)
            {
                throw new OptimizationException("The length of c must match the dimension of Q.");
            }

            if (InequalityMatrix != null && InequalityVector != null && InequalityMatrix.RowCount != InequalityVector.Count)
            {
                throw new OptimizationException("Inequality matrix rows must match the size of its right-hand side.");
            }

            if (EqualityMatrix != null && EqualityVector != null && EqualityMatrix.RowCount != EqualityVector.Count)
            {
                throw new OptimizationException("Equality matrix rows must match the size of its right-hand side.");
            }
        }
    }
}
