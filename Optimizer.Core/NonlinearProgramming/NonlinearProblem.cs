using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Represents a generic nonlinear optimisation problem with optional constraints.
    /// </summary>
    public sealed class NonlinearProblem
    {
        public NonlinearProblem(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IReadOnlyCollection<NonlinearConstraint> constraints,
            Vector<double> initialGuess,
            Vector<double> lowerBounds = null,
            Vector<double> upperBounds = null,
            Matrix<double> equalityMatrix = null,
            Vector<double> equalityVector = null,
            Matrix<double> inequalityMatrix = null,
            Vector<double> inequalityVector = null)
        {
            Objective = objective ?? throw new OptimizationException("An objective function must be supplied.");
            Gradient = gradient;
            Constraints = constraints ?? Array.Empty<NonlinearConstraint>();
            InitialGuess = initialGuess ?? throw new OptimizationException("An initial guess is required.");
            LowerBounds = lowerBounds;
            UpperBounds = upperBounds;
            EqualityMatrix = equalityMatrix;
            EqualityVector = equalityVector;
            InequalityMatrix = inequalityMatrix;
            InequalityVector = inequalityVector;
        }

        public Func<Vector<double>, double> Objective { get; }

        public Func<Vector<double>, Vector<double>> Gradient { get; }

        public IReadOnlyCollection<NonlinearConstraint> Constraints { get; }

        public Vector<double> InitialGuess { get; }

        public Vector<double> LowerBounds { get; }

        public Vector<double> UpperBounds { get; }

        public Matrix<double> EqualityMatrix { get; }

        public Vector<double> EqualityVector { get; }

        public Matrix<double> InequalityMatrix { get; }

        public Vector<double> InequalityVector { get; }
    }
}
