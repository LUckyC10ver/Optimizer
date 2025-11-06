using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    public static class RunSqp
    {
        public static Solution Solve(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IEnumerable<NonlinearConstraint> constraints,
            Vector<double> initialGuess,
            SqpOptions options = null,
            out SqpInfo info)
        {
            var solver = new SequentialQuadraticProgrammingSolver();
            return solver.Solve(objective, gradient, constraints, initialGuess, options, out info);
        }

        public static Solution Solve(
            ref Vector<double> xout,
            out SqpInfo info,
            out Vector<double> multipliers,
            out IReadOnlyList<int> activeSet,
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IEnumerable<NonlinearConstraint> constraints,
            Vector<double> lowerBounds,
            Vector<double> upperBounds,
            Matrix<double> linearEqualityMatrix,
            Vector<double> linearEqualityVector,
            Matrix<double> linearInequalityMatrix,
            Vector<double> linearInequalityVector,
            SqpOptions options = null,
            Matrix<double> initialHessian = null)
        {
            var solver = new SequentialQuadraticProgrammingSolver();
            var constraintList = constraints != null
                ? new List<NonlinearConstraint>(constraints)
                : new List<NonlinearConstraint>();

            var dimension = xout?.Count
                            ?? lowerBounds?.Count
                            ?? upperBounds?.Count
                            ?? linearEqualityMatrix?.ColumnCount
                            ?? linearInequalityMatrix?.ColumnCount
                            ?? throw new OptimizationException("Unable to infer the problem dimension for SQP.");

            var problem = new NonlinearProblem(
                objective,
                gradient,
                constraintList,
                xout ?? Vector<double>.Build.Dense(dimension),
                dimension,
                linearEqualityMatrix,
                linearEqualityVector,
                linearInequalityMatrix,
                linearInequalityVector,
                lowerBounds,
                upperBounds);

            var solution = solver.Solve(problem, options, out info, initialHessian, out multipliers, out activeSet);
            if (solution.OptimalPoint != null)
            {
                xout = solution.OptimalPoint.Clone();
            }

            return solution;
        }
    }
}
