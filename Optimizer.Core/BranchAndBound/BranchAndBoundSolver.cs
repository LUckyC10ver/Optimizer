using System;
using System.Collections.Generic;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;
using Optimizer.Core.LinearProgramming;

namespace Optimizer.Core.BranchAndBound
{
    /// <summary>
    /// Implements a depth-first branch-and-bound routine using the simplex-based linear solver as relaxation engine.
    /// </summary>
    public sealed class BranchAndBoundSolver
    {
        private readonly LinearProgrammingSolver _linearSolver = new LinearProgrammingSolver();

        public Solution Solve(LinearProblem relaxation, IEnumerable<int> integerIndices, SolverOptions options = null)
        {
            var problem = new MixedIntegerProblem(relaxation, integerIndices);
            return Solve(problem, options);
        }

        public Solution Solve(MixedIntegerProblem problem, SolverOptions options = null)
        {
            if (problem == null)
            {
                throw new ArgumentNullException(nameof(problem));
            }

            var tolerance = options?.Tolerance ?? 1e-6;
            var bestObjective = double.PositiveInfinity;
            Vector<double> bestSolution = null;
            var stack = new Stack<MixedIntegerProblem>();
            stack.Push(problem);
            var watch = Stopwatch.StartNew();
            var iterations = 0;

            while (stack.Count > 0)
            {
                if (options?.TimeLimit is { } limit && watch.Elapsed > limit)
                {
                    watch.Stop();
                    return new Solution(bestSolution, bestObjective, SolverResultStatus.TimeLimit, iterations, watch.Elapsed);
                }

                iterations++;
                var current = stack.Pop();
                var relaxationResult = _linearSolver.Solve(current.Relaxation, options);

                if (relaxationResult.Status != SolverResultStatus.Optimal)
                {
                    continue;
                }

                if (relaxationResult.OptimalValue >= bestObjective)
                {
                    continue;
                }

                var candidate = relaxationResult.OptimalX;
                var fractionalIndex = FindFractionalIndex(candidate, current.IntegerIndices, tolerance);

                if (fractionalIndex == -1)
                {
                    bestObjective = relaxationResult.OptimalValue;
                    bestSolution = candidate.Clone();
                    continue;
                }

                var value = candidate[fractionalIndex];
                var lower = Math.Floor(value);
                var upper = Math.Ceiling(value);

                var columns = current.Relaxation.A.ColumnCount;
                var leqCoefficients = Vector<double>.Build.Dense(columns);
                leqCoefficients[fractionalIndex] = 1.0;
                stack.Push(current.WithAdditionalConstraint(leqCoefficients, lower));

                var geqCoefficients = Vector<double>.Build.Dense(columns);
                geqCoefficients[fractionalIndex] = -1.0;
                stack.Push(current.WithAdditionalConstraint(geqCoefficients, -upper));
            }

            watch.Stop();

            if (double.IsPositiveInfinity(bestObjective))
            {
                return new Solution(Vector<double>.Build.Dense(problem.Relaxation.C.Count), double.NaN, SolverResultStatus.Infeasible, iterations, watch.Elapsed);
            }

            return new Solution(bestSolution, bestObjective, SolverResultStatus.Optimal, iterations, watch.Elapsed);
        }

        private static int FindFractionalIndex(Vector<double> solution, HashSet<int> integerIndices, double tolerance)
        {
            foreach (var index in integerIndices)
            {
                if (index < 0 || index >= solution.Count)
                {
                    continue;
                }

                var value = solution[index];
                var nearest = Math.Round(value);
                if (Math.Abs(value - nearest) > tolerance)
                {
                    return index;
                }
            }

            return -1;
        }
    }
}
