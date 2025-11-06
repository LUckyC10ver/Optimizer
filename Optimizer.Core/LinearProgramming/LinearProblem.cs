using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.LinearProgramming
{
    public class LinearProblem
    {
        public Matrix<double> A { get; }

        public Vector<double> B { get; }

        public Vector<double> C { get; }

        public bool IsMinimize { get; }

        public Vector<double> LowerBounds { get; }

        public Vector<double> UpperBounds { get; }

        public Matrix<double> EqualityMatrix { get; }

        public Vector<double> EqualityVector { get; }

        public IReadOnlyList<LinearConstraint> Constraints { get; }

        public LinearProblem(
            Matrix<double> a,
            Vector<double> bVector,
            Vector<double> cVector,
            bool isMinimize = true,
            IReadOnlyList<LinearConstraint> constraints = null,
            Vector<double> lowerBounds = null,
            Vector<double> upperBounds = null,
            Matrix<double> equalityMatrix = null,
            Vector<double> equalityVector = null)
        {
            A = a;
            B = bVector;
            C = cVector;
            IsMinimize = isMinimize;
            Constraints = constraints ?? new List<LinearConstraint>();
            LowerBounds = lowerBounds;
            UpperBounds = upperBounds;
            EqualityMatrix = equalityMatrix;
            EqualityVector = equalityVector;
        }
    }
}
