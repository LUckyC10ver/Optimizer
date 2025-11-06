using System;
using System.Collections.Generic;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;
using Optimizer.Core.LinearProgramming;

namespace Optimizer.Core.BranchAndBound
{
    public class BranchAndBoundSolver
    {
        private readonly LinearProgrammingSolver _linearSolver = new LinearProgrammingSolver();

        public Solution Solve(MixedIntegerProblem problem, SolverOptions options = null)
        {
            if (problem == null)
            {
                throw new OptimizationException("Mixed integer problem cannot be null.");
            }

            options ??= new SolverOptions();

            if (problem.IntegerIndices.Count == 0)
            {
                return _linearSolver.Solve(problem.BaseProblem, options);
            }

            var stopwatch = Stopwatch.StartNew();
            var tolerance = options.Tolerance > 0 ? options.Tolerance : 1e-6;
            var iterationLimit = options.MaxIterations;

            var variableCount = problem.VariableCount;
            if (variableCount == 0)
            {
                throw new OptimizationException("The mixed-integer problem must contain at least one variable.");
            }

            var initialLower = problem.GetInitialLowerBounds().ToArray();
            var initialUpper = problem.GetInitialUpperBounds().ToArray();

            ValidateBounds(initialLower, initialUpper, tolerance);

            var nodesProcessed = 0;
            var iterationLimitHit = false;
            var unboundedDetected = false;

            var bestSolution = default(Vector<double>);
            var bestValue = 0.0;
            var hasBest = false;

            var isMinimize = problem.BaseProblem.IsMinimize;

            var stack = new Stack<Node>();
            stack.Push(new Node(initialLower, initialUpper, depth: 0));

            while (stack.Count > 0)
            {
                if (iterationLimit > 0 && nodesProcessed >= iterationLimit)
                {
                    iterationLimitHit = true;
                    break;
                }

                var node = stack.Pop();
                nodesProcessed++;

                var relaxation = BuildRelaxation(problem.BaseProblem, node.LowerBounds, node.UpperBounds);
                Solution lpSolution;
                try
                {
                    lpSolution = _linearSolver.Solve(relaxation, options);
                }
                catch (OptimizationException)
                {
                    // Treat solver failures as node pruning.
                    continue;
                }

                if (lpSolution.Status == SolverResultStatus.Unbounded)
                {
                    unboundedDetected = true;
                    break;
                }

                if (lpSolution.Status != SolverResultStatus.Optimal)
                {
                    continue;
                }

                var nodeValue = lpSolution.OptimalValue;

                if (hasBest)
                {
                    if (isMinimize)
                    {
                        if (nodeValue >= bestValue - tolerance)
                        {
                            continue;
                        }
                    }
                    else
                    {
                        if (nodeValue <= bestValue + tolerance)
                        {
                            continue;
                        }
                    }
                }

                var solutionPoint = lpSolution.OptimalPoint;
                if (solutionPoint == null)
                {
                    continue;
                }

                var branchVariable = -1;
                var maxFraction = 0.0;

                foreach (var index in problem.IntegerIndices)
                {
                    var value = solutionPoint[index];
                    var fractional = Math.Abs(value - Math.Round(value));
                    if (fractional > tolerance && fractional > maxFraction)
                    {
                        maxFraction = fractional;
                        branchVariable = index;
                    }
                }

                if (branchVariable == -1)
                {
                    // Integer feasible solution found.
                    if (!hasBest || (isMinimize ? nodeValue < bestValue - tolerance : nodeValue > bestValue + tolerance))
                    {
                        bestSolution = solutionPoint.Clone();
                        bestValue = nodeValue;
                        hasBest = true;
                    }

                    continue;
                }

                var valueToBranch = solutionPoint[branchVariable];
                var floorValue = Math.Floor(valueToBranch);
                var ceilValue = Math.Ceiling(valueToBranch);

                // Left branch: x_i <= floor
                if (floorValue >= node.LowerBounds[branchVariable] - tolerance)
                {
                    var lower = (double[])node.LowerBounds.Clone();
                    var upper = (double[])node.UpperBounds.Clone();
                    upper[branchVariable] = Math.Min(upper[branchVariable], floorValue);

                    if (IsFeasibleBounds(lower[branchVariable], upper[branchVariable], tolerance))
                    {
                        stack.Push(new Node(lower, upper, node.Depth + 1));
                    }
                }

                // Right branch: x_i >= ceil
                if (ceilValue <= node.UpperBounds[branchVariable] + tolerance)
                {
                    var lower = (double[])node.LowerBounds.Clone();
                    var upper = (double[])node.UpperBounds.Clone();
                    lower[branchVariable] = Math.Max(lower[branchVariable], ceilValue);

                    if (IsFeasibleBounds(lower[branchVariable], upper[branchVariable], tolerance))
                    {
                        stack.Push(new Node(lower, upper, node.Depth + 1));
                    }
                }
            }

            stopwatch.Stop();

            if (unboundedDetected)
            {
                return new Solution
                {
                    Status = SolverResultStatus.Unbounded,
                    Iterations = nodesProcessed,
                    SolveTime = stopwatch.Elapsed,
                    Message = "LP relaxation detected an unbounded objective, so the mixed-integer problem is unbounded."
                };
            }

            if (hasBest)
            {
                return new Solution
                {
                    Status = SolverResultStatus.Optimal,
                    OptimalPoint = bestSolution,
                    OptimalValue = bestValue,
                    Iterations = nodesProcessed,
                    SolveTime = stopwatch.Elapsed,
                    Message = $"Explored {nodesProcessed} nodes in branch-and-bound search."
                };
            }

            return new Solution
            {
                Status = iterationLimitHit ? SolverResultStatus.IterationLimit : SolverResultStatus.Infeasible,
                Iterations = nodesProcessed,
                SolveTime = stopwatch.Elapsed,
                Message = iterationLimitHit
                    ? "Branch-and-bound iteration limit reached before finding an integer feasible solution."
                    : "No feasible integer solution found."
            };
        }

        public Solution Solve(LinearProblem problem, IEnumerable<int> integerIndices, SolverOptions options = null)
        {
            var mixedProblem = new MixedIntegerProblem(problem, integerIndices);
            return Solve(mixedProblem, options);
        }

        private static void ValidateBounds(IReadOnlyList<double> lower, IReadOnlyList<double> upper, double tolerance)
        {
            if (lower.Count != upper.Count)
            {
                throw new OptimizationException("Lower and upper bounds must have the same length.");
            }

            for (var i = 0; i < lower.Count; i++)
            {
                if (!IsFeasibleBounds(lower[i], upper[i], tolerance))
                {
                    throw new OptimizationException($"Inconsistent bounds for variable {i}: lower={lower[i]}, upper={upper[i]}.");
                }
            }
        }

        private static bool IsFeasibleBounds(double lower, double upper, double tolerance)
        {
            return lower <= upper + tolerance;
        }

        private static LinearProblem BuildRelaxation(LinearProblem baseProblem, double[] lowerBounds, double[] upperBounds)
        {
            var dimension = baseProblem.C.Count;
            var baseMatrix = baseProblem.A ?? Matrix<double>.Build.Dense(0, dimension);
            var baseVector = baseProblem.B ?? Vector<double>.Build.Dense(baseMatrix.RowCount);

            if (baseMatrix.RowCount != baseVector.Count)
            {
                throw new OptimizationException("Inequality matrix row count must match the vector length.");
            }

            var boundRows = 0;
            for (var i = 0; i < dimension; i++)
            {
                if (!double.IsNegativeInfinity(lowerBounds[i]))
                {
                    boundRows++;
                }

                if (!double.IsPositiveInfinity(upperBounds[i]))
                {
                    boundRows++;
                }
            }

            var augmentedRows = baseMatrix.RowCount + boundRows;
            var augmentedMatrix = Matrix<double>.Build.Dense(augmentedRows, dimension);
            var augmentedVector = Vector<double>.Build.Dense(augmentedRows);

            for (var row = 0; row < baseMatrix.RowCount; row++)
            {
                augmentedVector[row] = baseVector[row];
                for (var column = 0; column < dimension; column++)
                {
                    augmentedMatrix[row, column] = baseMatrix[row, column];
                }
            }

            var currentRow = baseMatrix.RowCount;
            for (var i = 0; i < dimension; i++)
            {
                if (!double.IsNegativeInfinity(lowerBounds[i]))
                {
                    augmentedMatrix[currentRow, i] = -1.0;
                    augmentedVector[currentRow] = -lowerBounds[i];
                    currentRow++;
                }

                if (!double.IsPositiveInfinity(upperBounds[i]))
                {
                    augmentedMatrix[currentRow, i] = 1.0;
                    augmentedVector[currentRow] = upperBounds[i];
                    currentRow++;
                }
            }

            var equalityMatrix = baseProblem.EqualityMatrix ?? Matrix<double>.Build.Dense(0, dimension);
            var equalityVector = baseProblem.EqualityVector ?? Vector<double>.Build.Dense(equalityMatrix.RowCount);

            if (equalityMatrix.ColumnCount != dimension)
            {
                throw new OptimizationException("Equality matrix must have the same number of columns as the objective vector.");
            }

            if (equalityMatrix.RowCount != equalityVector.Count)
            {
                throw new OptimizationException("Equality matrix row count must match the equality vector length.");
            }

            return new LinearProblem(
                augmentedMatrix,
                augmentedVector,
                baseProblem.C,
                baseProblem.IsMinimize,
                lowerBounds: Vector<double>.Build.DenseOfArray(lowerBounds),
                upperBounds: Vector<double>.Build.DenseOfArray(upperBounds),
                equalityMatrix: equalityMatrix,
                equalityVector: equalityVector);
        }

        private readonly struct Node
        {
            public Node(double[] lowerBounds, double[] upperBounds, int depth)
            {
                LowerBounds = lowerBounds;
                UpperBounds = upperBounds;
                Depth = depth;
            }

            public double[] LowerBounds { get; }

            public double[] UpperBounds { get; }

            public int Depth { get; }
        }
    }
}
