using System;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.LinearProgramming
{
    /// <summary>
    /// Basic implementation of the simplex algorithm for dense problems.
    /// </summary>
    public sealed class LinearProgrammingSolver
    {
        public Solution Solve(double[,] a, double[] b, double[] c, SolverOptions options = null)
        {
            var matrixA = Matrix<double>.Build.DenseOfArray(a ?? throw new OptimizationException("Matrix A cannot be null."));
            var vectorB = Vector<double>.Build.DenseOfArray(b ?? throw new OptimizationException("Vector b cannot be null."));
            var vectorC = Vector<double>.Build.DenseOfArray(c ?? throw new OptimizationException("Vector c cannot be null."));
            var problem = new LinearProblem(matrixA, vectorB, vectorC);
            return Solve(problem, options);
        }

        public Solution Solve(LinearProblem problem, SolverOptions options = null)
        {
            if (problem == null)
            {
                throw new ArgumentNullException(nameof(problem));
            }

            problem.Validate();

            var tolerance = options?.Tolerance ?? 1e-8;
            var maxIterations = options?.MaxIterations ?? 10_000;

            var watch = Stopwatch.StartNew();
            var result = ExecuteSimplex(problem, tolerance, maxIterations);
            watch.Stop();

            return new Solution(result.Solution, result.ObjectiveValue, result.Status, result.Iterations, watch.Elapsed);
        }

        private (Vector<double> Solution, double ObjectiveValue, SolverResultStatus Status, int Iterations) ExecuteSimplex(
            LinearProblem problem,
            double tolerance,
            int maxIterations)
        {
            var m = problem.A.RowCount;
            var n = problem.A.ColumnCount;

            var tableau = Matrix<double>.Build.Dense(m + 1, n + m + 1);

            // Build tableau with slack variables
            for (var i = 0; i < m; i++)
            {
                tableau[i, 0] = problem.B[i];
                for (var j = 0; j < n; j++)
                {
                    tableau[i, j + 1] = problem.A[i, j];
                }
                tableau[i, n + 1 + i] = 1.0;
            }

            // Objective row (minimisation -> convert to maximisation by negating coefficients)
            tableau[m, 0] = 0.0;
            for (var j = 0; j < n; j++)
            {
                var coefficient = problem.IsMinimisation ? -problem.C[j] : problem.C[j];
                tableau[m, j + 1] = coefficient;
            }

            var basis = new int[m];
            for (var i = 0; i < m; i++)
            {
                basis[i] = n + i;
            }

            var iterations = 0;

            while (iterations < maxIterations)
            {
                iterations++;

                var pivotColumn = -1;
                var maxValue = tolerance;
                for (var j = 1; j < tableau.ColumnCount; j++)
                {
                    if (tableau[m, j] > maxValue)
                    {
                        maxValue = tableau[m, j];
                        pivotColumn = j;
                    }
                }

                if (pivotColumn == -1)
                {
                    // optimal
                    var solution = Vector<double>.Build.Dense(n);
                    for (var i = 0; i < m; i++)
                    {
                        if (basis[i] < n)
                        {
                            solution[basis[i]] = tableau[i, 0];
                        }
                    }

                    var objective = problem.IsMinimisation ? tableau[m, 0] : -tableau[m, 0];
                    return (solution, objective, SolverResultStatus.Optimal, iterations);
                }

                var pivotRow = -1;
                double bestRatio = double.PositiveInfinity;
                for (var i = 0; i < m; i++)
                {
                    var value = tableau[i, pivotColumn];
                    if (value > tolerance)
                    {
                        var ratio = tableau[i, 0] / value;
                        if (ratio < bestRatio)
                        {
                            bestRatio = ratio;
                            pivotRow = i;
                        }
                    }
                }

                if (pivotRow == -1)
                {
                    return (Vector<double>.Build.Dense(n), double.NegativeInfinity, SolverResultStatus.Unbounded, iterations);
                }

                Pivot(tableau, pivotRow, pivotColumn);
                basis[pivotRow] = pivotColumn - 1;
            }

            return (Vector<double>.Build.Dense(n), tableau[m, 0], SolverResultStatus.IterationLimit, iterations);
        }

        private static void Pivot(Matrix<double> tableau, int pivotRow, int pivotColumn)
        {
            var pivot = tableau[pivotRow, pivotColumn];
            var columnCount = tableau.ColumnCount;
            var rowCount = tableau.RowCount;

            for (var j = 0; j < columnCount; j++)
            {
                tableau[pivotRow, j] /= pivot;
            }

            for (var i = 0; i < rowCount; i++)
            {
                if (i == pivotRow)
                {
                    continue;
                }

                var factor = tableau[i, pivotColumn];
                if (Math.Abs(factor) < 1e-16)
                {
                    continue;
                }

                for (var j = 0; j < columnCount; j++)
                {
                    tableau[i, j] -= factor * tableau[pivotRow, j];
                }
            }
        }
    }
}
