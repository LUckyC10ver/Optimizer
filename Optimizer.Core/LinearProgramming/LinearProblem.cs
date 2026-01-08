using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.LinearProgramming
{
    /// <summary>
    /// Represents a linear programming model in canonical form.
    /// </summary>
    public sealed class LinearProblem
    {
        public LinearProblem(Matrix<double> a, Vector<double> b, Vector<double> c, bool isMinimisation = true)
        {
            A = a;
            B = b;
            C = c;
            IsMinimisation = isMinimisation;
        }

        public Matrix<double> A { get; }

        public Vector<double> B { get; }

        public Vector<double> C { get; }

        public bool IsMinimisation { get; }

        public void Validate()
        {
            if (A == null)
            {
                throw new OptimizationException("Constraint matrix A cannot be null.");
            }

            if (B == null)
            {
                throw new OptimizationException("Right-hand side vector b cannot be null.");
            }

            if (C == null)
            {
                throw new OptimizationException("Objective vector c cannot be null.");
            }

            if (A.RowCount != B.Count)
            {
                throw new OptimizationException("The number of rows in A must match the length of b.");
            }

            if (A.ColumnCount != C.Count)
            {
                throw new OptimizationException("The number of columns in A must match the length of c.");
            }
        }
    }
}
