using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;
using Optimizer.Core.LinearProgramming;

namespace Optimizer.Core.QuadraticProgramming
{
    public class QuadraticProblem
    {
        public Matrix<double> Q { get; }

        public Vector<double> C { get; }

        public Matrix<double> A { get; }

        public Vector<double> B { get; }

        public Matrix<double> EqualityMatrix { get; }

        public Vector<double> EqualityVector { get; }

        public Vector<double> LowerBounds { get; }

        public Vector<double> UpperBounds { get; }

        public Vector<double> InitialGuess { get; }

        public bool IsMinimize { get; }

        public IReadOnlyList<LinearConstraint> Constraints { get; }

        public QuadraticProblem(
            Matrix<double> q,
            Vector<double> c,
            Matrix<double> inequalityMatrix = null,
            Vector<double> inequalityVector = null,
            bool isMinimize = true,
            Matrix<double> equalityMatrix = null,
            Vector<double> equalityVector = null,
            Vector<double> lowerBounds = null,
            Vector<double> upperBounds = null,
            Vector<double> initialGuess = null,
            IReadOnlyList<LinearConstraint> constraints = null)
        {
            Q = q ?? throw new OptimizationException("Quadratic matrix Q cannot be null.");
            C = c ?? throw new OptimizationException("Linear term vector c cannot be null.");
            A = inequalityMatrix;
            B = inequalityVector;
            EqualityMatrix = equalityMatrix;
            EqualityVector = equalityVector;
            LowerBounds = lowerBounds;
            UpperBounds = upperBounds;
            InitialGuess = initialGuess;
            IsMinimize = isMinimize;
            Constraints = constraints ?? new List<LinearConstraint>();

            ValidateDimensions();
        }

        private void ValidateDimensions()
        {
            var dimension = Q.ColumnCount;

            if (Q.RowCount != dimension)
            {
                throw new OptimizationException("Matrix Q must be square.");
            }

            if (C.Count != dimension)
            {
                throw new OptimizationException("Vector c must match the dimension of Q.");
            }

            if (A != null)
            {
                if (B == null)
                {
                    throw new OptimizationException("Inequality matrix A is provided but vector b is null.");
                }

                if (A.ColumnCount != dimension)
                {
                    throw new OptimizationException("Inequality matrix A must have the same number of columns as the problem dimension.");
                }

                if (A.RowCount != B.Count)
                {
                    throw new OptimizationException("Inequality matrix row count must match the length of vector b.");
                }
            }
            else if (B != null)
            {
                throw new OptimizationException("Inequality vector b is provided without a corresponding matrix A.");
            }

            if (EqualityMatrix != null)
            {
                if (EqualityVector == null)
                {
                    throw new OptimizationException("Equality matrix is provided but the equality vector is null.");
                }

                if (EqualityMatrix.ColumnCount != dimension)
                {
                    throw new OptimizationException("Equality matrix must have the same number of columns as the problem dimension.");
                }

                if (EqualityMatrix.RowCount != EqualityVector.Count)
                {
                    throw new OptimizationException("Equality matrix row count must match the length of the equality vector.");
                }
            }
            else if (EqualityVector != null)
            {
                throw new OptimizationException("Equality vector is provided without a corresponding matrix.");
            }

            if (LowerBounds != null && LowerBounds.Count != dimension)
            {
                throw new OptimizationException("Lower bounds vector must match the problem dimension.");
            }

            if (UpperBounds != null && UpperBounds.Count != dimension)
            {
                throw new OptimizationException("Upper bounds vector must match the problem dimension.");
            }

            if (LowerBounds != null && UpperBounds != null)
            {
                for (var i = 0; i < dimension; i++)
                {
                    if (UpperBounds[i] < LowerBounds[i])
                    {
                        throw new OptimizationException($"Upper bound at index {i} is smaller than the lower bound.");
                    }
                }
            }

            if (InitialGuess != null && InitialGuess.Count != dimension)
            {
                throw new OptimizationException("Initial guess vector must match the problem dimension.");
            }
        }
    }
}
