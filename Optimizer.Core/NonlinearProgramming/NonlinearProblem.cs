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

        public NonlinearProblem(Func<Vector<double>, double> objective, Func<Vector<double>, Vector<double>> gradient, IReadOnlyList<NonlinearConstraint> constraints, Vector<double> initialGuess, int variableCount)
        {
            Objective = objective;
            Gradient = gradient;
            Constraints = constraints ?? new List<NonlinearConstraint>();
            InitialGuess = initialGuess;
            VariableCount = variableCount;
        }
    }
}
