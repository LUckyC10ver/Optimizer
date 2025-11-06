using System;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.QuadraticProgramming
{
    /// <summary>
    /// Implements a simple projected gradient descent on a penalised quadratic objective.
    /// The routine is designed to provide a reasonable default while keeping the implementation compact.
    /// </summary>
    public sealed class QuadraticProgrammingSolver
    {
        public Solution Solve(QuadraticProblem problem, SolverOptions options = null)
        {
            if (problem == null)
            {
                throw new ArgumentNullException(nameof(problem));
            }

            problem.Validate();

            var tolerance = options?.Tolerance ?? 1e-6;
            var maxIterations = options?.MaxIterations ?? 5_000;
            var penaltyWeight = 100.0;
            var stepSize = 1.0 / Math.Max(problem.Q.RowCount, 1);

            var dimension = problem.Q.ColumnCount;
            var x = problem.InitialGuess?.Clone() ?? Vector<double>.Build.Dense(dimension);

            if (problem.LowerBounds != null)
            {
                for (var i = 0; i < dimension && i < problem.LowerBounds.Count; i++)
                {
                    x[i] = Math.Max(x[i], problem.LowerBounds[i]);
                }
            }

            if (problem.UpperBounds != null)
            {
                for (var i = 0; i < dimension && i < problem.UpperBounds.Count; i++)
                {
                    x[i] = Math.Min(x[i], problem.UpperBounds[i]);
                }
            }

            var gradient = Vector<double>.Build.Dense(dimension);
            var watch = Stopwatch.StartNew();
            var iterations = 0;

            for (; iterations < maxIterations; iterations++)
            {
                ComputeGradient(problem, x, penaltyWeight, gradient, out var violation);

                if (violation < tolerance && gradient.L2Norm() < tolerance)
                {
                    break;
                }

                x -= gradient * stepSize;
                Project(problem, x);
            }

            watch.Stop();

            var objective = EvaluateObjective(problem, x);
            if (!problem.IsMinimisation)
            {
                objective = -objective;
            }

            return new Solution(x.Clone(), objective, iterations < maxIterations ? SolverResultStatus.Optimal : SolverResultStatus.IterationLimit, iterations, watch.Elapsed);
        }

        private static void ComputeGradient(
            QuadraticProblem problem,
            Vector<double> point,
            double penalty,
            Vector<double> gradient,
            out double violation)
        {
            gradient.SetSubVector(0, gradient.Count, problem.Q * point + problem.C);
            violation = 0.0;

            if (problem.EqualityMatrix != null && problem.EqualityVector != null)
            {
                var residual = problem.EqualityMatrix * point - problem.EqualityVector;
                violation = Math.Max(violation, residual.L2Norm());
                gradient += problem.EqualityMatrix.TransposeThisAndMultiply(residual) * (2.0 * penalty);
            }

            if (problem.InequalityMatrix != null && problem.InequalityVector != null)
            {
                var residual = problem.InequalityMatrix * point - problem.InequalityVector;
                for (var i = 0; i < residual.Count; i++)
                {
                    if (residual[i] > 0.0)
                    {
                        violation = Math.Max(violation, residual[i]);
                        gradient += problem.InequalityMatrix.Row(i) * (2.0 * penalty * residual[i]);
                    }
                }
            }
        }

        private static double EvaluateObjective(QuadraticProblem problem, Vector<double> x)
        {
            var quadratic = 0.5 * x.DotProduct(problem.Q * x);
            var linear = problem.C.DotProduct(x);
            return quadratic + linear;
        }

        private static void Project(QuadraticProblem problem, Vector<double> vector)
        {
            if (problem.LowerBounds != null)
            {
                for (var i = 0; i < vector.Count && i < problem.LowerBounds.Count; i++)
                {
                    vector[i] = Math.Max(vector[i], problem.LowerBounds[i]);
                }
            }

            if (problem.UpperBounds != null)
            {
                for (var i = 0; i < vector.Count && i < problem.UpperBounds.Count; i++)
                {
                    vector[i] = Math.Min(vector[i], problem.UpperBounds[i]);
                }
            }
        }
    }
}
