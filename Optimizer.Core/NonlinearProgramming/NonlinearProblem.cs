using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Optimizer.Core.NonlinearProgramming
{
    public class NonlinearProblem
    {
        public Func<Vector<double>, double> Objective { get; }

        public Func<Vector<double>, Vector<double>> Gradient { get; }

        public IReadOnlyList<NonlinearConstraint> Constraints { get; }

        public Vector<double> InitialGuess { get; }

        public int VariableCount { get; }

        public Matrix<double> LinearEqualityMatrix { get; }

        public Vector<double> LinearEqualityVector { get; }

        public Matrix<double> LinearInequalityMatrix { get; }

        public Vector<double> LinearInequalityVector { get; }

        public Vector<double> LowerBounds { get; }

        public Vector<double> UpperBounds { get; }

        public NonlinearProblem(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IReadOnlyList<NonlinearConstraint> constraints,
            Vector<double> initialGuess,
            int variableCount,
            Matrix<double> linearEqualityMatrix = null,
            Vector<double> linearEqualityVector = null,
            Matrix<double> linearInequalityMatrix = null,
            Vector<double> linearInequalityVector = null,
            Vector<double> lowerBounds = null,
            Vector<double> upperBounds = null)
        {
            Objective = objective;
            Gradient = gradient;
            Constraints = constraints ?? new List<NonlinearConstraint>();
            InitialGuess = initialGuess;
            VariableCount = variableCount;
            LinearEqualityMatrix = linearEqualityMatrix;
            LinearEqualityVector = linearEqualityVector;
            LinearInequalityMatrix = linearInequalityMatrix;
            LinearInequalityVector = linearInequalityVector;
            LowerBounds = lowerBounds;
            UpperBounds = upperBounds;
        }
    }
}
