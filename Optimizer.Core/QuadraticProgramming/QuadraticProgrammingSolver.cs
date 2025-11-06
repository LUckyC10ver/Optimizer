using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;
using Optimizer.Core.LinearProgramming;

namespace Optimizer.Core.QuadraticProgramming
{
    public class QuadraticProgrammingSolver
    {
        public Solution Solve(QuadraticProblem problem, SolverOptions options = null)
        {
            if (problem == null)
            {
                throw new OptimizationException("Quadratic programming problem cannot be null.");
            }

            options ??= new SolverOptions();
            var tolerance = options.Tolerance > 0 ? options.Tolerance : 1e-8;
            var constraintTolerance = options.ConstraintTolerance > 0 ? options.ConstraintTolerance : tolerance * 10;
            var maxIterations = options.MaxIterations > 0 ? options.MaxIterations : 250;

            var originalQ = problem.Q;
            var originalC = problem.C;

            if (originalQ.ColumnCount != originalQ.RowCount)
            {
                throw new OptimizationException("Matrix Q must be square.");
            }

            var dimension = originalC.Count;

            if (originalQ.ColumnCount != dimension)
            {
                throw new OptimizationException("Matrix Q must match the dimension of vector c.");
            }

            var workingQ = problem.IsMinimize ? originalQ.Clone() : originalQ.Negate();
            var workingC = problem.IsMinimize ? originalC.Clone() : originalC.Negate();

            var inequalityMatrix = problem.A ?? Matrix<double>.Build.Dense(0, dimension);
            var inequalityVector = problem.B ?? Vector<double>.Build.Dense(inequalityMatrix.RowCount);

            if (inequalityMatrix.RowCount != inequalityVector.Count)
            {
                throw new OptimizationException("Inequality constraint matrix row count must match the vector length.");
            }

            if (inequalityMatrix.ColumnCount != dimension)
            {
                throw new OptimizationException("Inequality constraint matrix must have the same number of columns as vector c.");
            }

            var equalityMatrix = problem.EqualityMatrix ?? Matrix<double>.Build.Dense(0, dimension);
            var equalityVector = problem.EqualityVector ?? Vector<double>.Build.Dense(equalityMatrix.RowCount);

            if (equalityMatrix.RowCount != equalityVector.Count)
            {
                throw new OptimizationException("Equality constraint matrix row count must match the equality vector length.");
            }

            if (equalityMatrix.ColumnCount != dimension)
            {
                throw new OptimizationException("Equality constraint matrix must have the same number of columns as vector c.");
            }

            var lowerBounds = problem.LowerBounds;
            var upperBounds = problem.UpperBounds;

            var feasibilityPoint = GetInitialPoint(problem, inequalityMatrix, inequalityVector, equalityMatrix, equalityVector, tolerance, constraintTolerance);
            var x = feasibilityPoint.Clone();

            var allConstraints = BuildConstraints(inequalityMatrix, inequalityVector, lowerBounds, upperBounds);

            var activeSet = InitialiseActiveSet(allConstraints, equalityMatrix, equalityVector, x, constraintTolerance);

            var solution = new Solution
            {
                Status = SolverResultStatus.Unknown,
                OptimalPoint = x.Clone(),
                OptimalValue = EvaluateObjective(originalQ, originalC, x)
            };

            var qpIterations = 0;

            while (qpIterations < maxIterations)
            {
                qpIterations++;

                var gradient = workingQ * x + workingC;

                var activeMatrix = BuildActiveMatrix(equalityMatrix, allConstraints, activeSet, dimension);

                Vector<double> searchDirection;
                Vector<double> multipliers;
                try
                {
                    (searchDirection, multipliers) = SolveKktSystem(workingQ, gradient, activeMatrix, tolerance);
                }
                catch (Exception ex)
                {
                    throw new OptimizationException($"Failed to solve KKT system: {ex.Message}");
                }

                if (searchDirection.L2Norm() <= tolerance)
                {
                    var lambdaIndex = equalityMatrix.RowCount;
                    var mostNegative = 0.0;
                    var constraintToRemove = -1;

                    foreach (var active in activeSet)
                    {
                        if (active.IsEquality)
                        {
                            continue;
                        }

                        var multiplier = multipliers[lambdaIndex++];
                        if (multiplier < mostNegative - tolerance)
                        {
                            mostNegative = multiplier;
                            constraintToRemove = active.Index;
                        }
                    }

                    if (constraintToRemove < 0)
                    {
                        solution.OptimalPoint = x.Clone();
                        solution.OptimalValue = EvaluateObjective(originalQ, originalC, x);
                        solution.Iterations = qpIterations;
                        solution.Status = SolverResultStatus.Optimal;
                        solution.Message = BuildActiveSetMessage(activeSet);
                        return solution;
                    }

                    activeSet.RemoveAll(cn => cn.Index == constraintToRemove);
                    continue;
                }

                var (stepLength, blockingConstraint) = ComputeStepLength(x, searchDirection, allConstraints, activeSet, constraintTolerance);
                x = x + stepLength * searchDirection;

                if (blockingConstraint != null)
                {
                    activeSet.Add(blockingConstraint);
                }
            }

            solution.OptimalPoint = x.Clone();
            solution.OptimalValue = EvaluateObjective(originalQ, originalC, x);
            solution.Iterations = qpIterations;
            solution.Status = SolverResultStatus.IterationLimit;
            solution.Message = "Maximum iterations reached before convergence.";
            return solution;
        }

        public Solution Solve(
            Vector<double> initialGuess,
            Matrix<double> q,
            Vector<double> c,
            Matrix<double> inequalityMatrix,
            Vector<double> inequalityVector,
            SolverOptions options = null,
            Matrix<double> equalityMatrix = null,
            Vector<double> equalityVector = null,
            Vector<double> lowerBounds = null,
            Vector<double> upperBounds = null)
        {
            var problem = new QuadraticProblem(
                q ?? throw new OptimizationException("Matrix Q cannot be null."),
                c ?? throw new OptimizationException("Vector c cannot be null."),
                inequalityMatrix,
                inequalityVector,
                true,
                equalityMatrix,
                equalityVector,
                lowerBounds,
                upperBounds,
                initialGuess);
            return Solve(problem, options);
        }

        private static string BuildActiveSetMessage(IReadOnlyCollection<ConstraintBinding> activeSet)
        {
            if (activeSet.Count == 0)
            {
                return string.Empty;
            }

            var indices = new List<string>();
            foreach (var constraint in activeSet)
            {
                string prefix;
                var index = constraint.OriginalIndex;
                if (constraint.IsEquality)
                {
                    prefix = "eq";
                }
                else if (constraint.Source == ConstraintSource.Inequality)
                {
                    prefix = "ieq";
                }
                else if (constraint.Source == ConstraintSource.LowerBound)
                {
                    prefix = "lbd";
                }
                else
                {
                    prefix = "ubd";
                }

                indices.Add($"{prefix}:{index}");
            }

            return $"Active constraints: {string.Join(", ", indices)}";
        }

        private static Vector<double> GetInitialPoint(
            QuadraticProblem problem,
            Matrix<double> inequalityMatrix,
            Vector<double> inequalityVector,
            Matrix<double> equalityMatrix,
            Vector<double> equalityVector,
            double tolerance,
            double constraintTolerance)
        {
            if (problem.InitialGuess != null)
            {
                var guess = problem.InitialGuess.Clone();
                if (IsFeasible(guess, inequalityMatrix, inequalityVector, equalityMatrix, equalityVector, problem.LowerBounds, problem.UpperBounds, constraintTolerance))
                {
                    return guess;
                }
            }

            if ((inequalityMatrix.RowCount == 0 || inequalityVector.Count == 0) && equalityMatrix.RowCount == 0 && problem.LowerBounds == null && problem.UpperBounds == null)
            {
                return Vector<double>.Build.Dense(problem.C.Count);
            }

            var lpSolver = new LinearProgrammingSolver();
            var zeroObjective = new double[problem.C.Count];

            var lpMatrix = inequalityMatrix;
            var lpVector = inequalityVector;
            var (boundMatrix, boundVector) = BuildBoundConstraints(problem.LowerBounds, problem.UpperBounds, problem.C.Count);

            if (boundMatrix.RowCount > 0)
            {
                lpMatrix = lpMatrix.Stack(boundMatrix);
                lpVector = Vector<double>.Build.DenseOfEnumerable(lpVector.Concat(boundVector));
            }

            var inequalityArray = ToArray(lpMatrix);
            var inequalityVectorArray = lpVector.Count > 0 ? lpVector.ToArray() : Array.Empty<double>();
            var equalityArray = equalityMatrix.RowCount > 0 ? ToArray(equalityMatrix) : new double[0, equalityMatrix.ColumnCount];
            var equalityVectorArray = equalityVector.Count > 0 ? equalityVector.ToArray() : Array.Empty<double>();

            var solution = lpSolver.Solve(inequalityArray, inequalityVectorArray, zeroObjective, equalityArray, equalityVectorArray);
            if (solution.Status != SolverResultStatus.Optimal)
            {
                throw new OptimizationException("Failed to locate a feasible starting point for the quadratic program.");
            }

            return solution.OptimalPoint;
        }

        private static (Matrix<double> Matrix, Vector<double> Vector) BuildBoundConstraints(Vector<double> lowerBounds, Vector<double> upperBounds, int dimension)
        {
            var rows = new List<Vector<double>>();
            var rhs = new List<double>();

            if (lowerBounds != null)
            {
                for (var i = 0; i < dimension; i++)
                {
                    var lower = lowerBounds[i];
                    if (double.IsNegativeInfinity(lower))
                    {
                        continue;
                    }

                    var row = Vector<double>.Build.Dense(dimension);
                    row[i] = -1.0;
                    rows.Add(row);
                    rhs.Add(-lower);
                }
            }

            if (upperBounds != null)
            {
                for (var i = 0; i < dimension; i++)
                {
                    var upper = upperBounds[i];
                    if (double.IsPositiveInfinity(upper))
                    {
                        continue;
                    }

                    var row = Vector<double>.Build.Dense(dimension);
                    row[i] = 1.0;
                    rows.Add(row);
                    rhs.Add(upper);
                }
            }

            if (rows.Count == 0)
            {
                return (Matrix<double>.Build.Dense(0, dimension), Vector<double>.Build.Dense(0));
            }

            var matrix = Matrix<double>.Build.DenseOfRowVectors(rows.ToArray());
            var vector = Vector<double>.Build.DenseOfEnumerable(rhs);
            return (matrix, vector);
        }

        private static bool IsFeasible(
            Vector<double> point,
            Matrix<double> inequalityMatrix,
            Vector<double> inequalityVector,
            Matrix<double> equalityMatrix,
            Vector<double> equalityVector,
            Vector<double> lowerBounds,
            Vector<double> upperBounds,
            double tolerance)
        {
            if (equalityMatrix.RowCount > 0)
            {
                var equalities = equalityMatrix * point - equalityVector;
                if (equalities.AbsoluteMaximum() > tolerance)
                {
                    return false;
                }
            }

            if (inequalityMatrix.RowCount > 0)
            {
                var inequalities = inequalityMatrix * point - inequalityVector;
                if (inequalities.Maximum() > tolerance)
                {
                    return false;
                }
            }

            if (lowerBounds != null)
            {
                for (var i = 0; i < lowerBounds.Count; i++)
                {
                    if (point[i] < lowerBounds[i] - tolerance)
                    {
                        return false;
                    }
                }
            }

            if (upperBounds != null)
            {
                for (var i = 0; i < upperBounds.Count; i++)
                {
                    if (point[i] > upperBounds[i] + tolerance)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        private static List<ConstraintBinding> BuildConstraints(
            Matrix<double> inequalityMatrix,
            Vector<double> inequalityVector,
            Vector<double> lowerBounds,
            Vector<double> upperBounds)
        {
            var constraints = new List<ConstraintBinding>();

            if (inequalityMatrix.RowCount > 0)
            {
                for (var i = 0; i < inequalityMatrix.RowCount; i++)
                {
                    constraints.Add(new ConstraintBinding
                    {
                        Index = constraints.Count,
                        OriginalIndex = i,
                        Source = ConstraintSource.Inequality,
                        Coefficients = inequalityMatrix.Row(i).Clone(),
                        RightHandSide = inequalityVector[i],
                        IsEquality = false
                    });
                }
            }

            if (lowerBounds != null)
            {
                for (var i = 0; i < lowerBounds.Count; i++)
                {
                    if (double.IsNegativeInfinity(lowerBounds[i]))
                    {
                        continue;
                    }

                    var row = Vector<double>.Build.Dense(lowerBounds.Count);
                    row[i] = -1.0;
                    constraints.Add(new ConstraintBinding
                    {
                        Index = constraints.Count,
                        OriginalIndex = i,
                        Source = ConstraintSource.LowerBound,
                        Coefficients = row,
                        RightHandSide = -lowerBounds[i],
                        IsEquality = false
                    });
                }
            }

            if (upperBounds != null)
            {
                for (var i = 0; i < upperBounds.Count; i++)
                {
                    if (double.IsPositiveInfinity(upperBounds[i]))
                    {
                        continue;
                    }

                    var row = Vector<double>.Build.Dense(upperBounds.Count);
                    row[i] = 1.0;
                    constraints.Add(new ConstraintBinding
                    {
                        Index = constraints.Count,
                        OriginalIndex = i,
                        Source = ConstraintSource.UpperBound,
                        Coefficients = row,
                        RightHandSide = upperBounds[i],
                        IsEquality = false
                    });
                }
            }

            return constraints;
        }

        private static List<ConstraintBinding> InitialiseActiveSet(
            IReadOnlyList<ConstraintBinding> inequalities,
            Matrix<double> equalityMatrix,
            Vector<double> equalityVector,
            Vector<double> point,
            double tolerance)
        {
            var activeSet = new List<ConstraintBinding>();

            if (equalityMatrix.RowCount > 0)
            {
                for (var i = 0; i < equalityMatrix.RowCount; i++)
                {
                    activeSet.Add(new ConstraintBinding
                    {
                        Index = -(i + 1),
                        OriginalIndex = i,
                        Source = ConstraintSource.Equality,
                        Coefficients = equalityMatrix.Row(i).Clone(),
                        RightHandSide = equalityVector[i],
                        IsEquality = true
                    });
                }
            }

            foreach (var constraint in inequalities)
            {
                var value = constraint.Coefficients.DotProduct(point) - constraint.RightHandSide;
                if (Math.Abs(value) <= tolerance)
                {
                    activeSet.Add(constraint);
                }
            }

            return activeSet;
        }

        private static (Vector<double> Direction, Vector<double> Multipliers) SolveKktSystem(
            Matrix<double> q,
            Vector<double> gradient,
            Matrix<double> activeMatrix,
            double tolerance)
        {
            var n = q.RowCount;
            var activeCount = activeMatrix.RowCount;

            if (activeCount == 0)
            {
                var cholesky = q.Cholesky();
                var direction = cholesky.Solve(-gradient);
                return (direction, Vector<double>.Build.Dense(0));
            }

            var size = n + activeCount;
            var kkt = Matrix<double>.Build.Dense(size, size);

            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    kkt[i, j] = q[i, j];
                }
            }

            var activeTranspose = activeMatrix.Transpose();
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < activeCount; j++)
                {
                    var value = activeTranspose[i, j];
                    kkt[i, n + j] = value;
                    kkt[n + j, i] = value;
                }
            }

            var rhs = Vector<double>.Build.Dense(size);
            for (var i = 0; i < n; i++)
            {
                rhs[i] = -gradient[i];
            }

            var solution = kkt.Solve(rhs);
            var directionVector = solution.SubVector(0, n);
            var multipliers = solution.SubVector(n, activeCount);
            return (directionVector, multipliers);
        }

        private static Matrix<double> BuildActiveMatrix(
            Matrix<double> equalityMatrix,
            IReadOnlyList<ConstraintBinding> inequalities,
            IReadOnlyCollection<ConstraintBinding> activeSet,
            int dimension)
        {
            if (activeSet.Count == 0 && equalityMatrix.RowCount == 0)
            {
                return Matrix<double>.Build.Dense(0, dimension);
            }

            var rows = new List<Vector<double>>();

            if (equalityMatrix.RowCount > 0)
            {
                for (var i = 0; i < equalityMatrix.RowCount; i++)
                {
                    rows.Add(equalityMatrix.Row(i).Clone());
                }
            }

            foreach (var constraint in activeSet)
            {
                if (constraint.IsEquality)
                {
                    continue;
                }

                rows.Add(constraint.Coefficients.Clone());
            }

            return rows.Count == 0
                ? Matrix<double>.Build.Dense(0, dimension)
                : Matrix<double>.Build.DenseOfRowVectors(rows.ToArray());
        }

        private static (double Step, ConstraintBinding BlockingConstraint) ComputeStepLength(
            Vector<double> point,
            Vector<double> direction,
            IReadOnlyList<ConstraintBinding> constraints,
            IReadOnlyCollection<ConstraintBinding> activeSet,
            double tolerance)
        {
            var step = 1.0;
            ConstraintBinding blockingConstraint = null;

            var activeLookup = new HashSet<int>(activeSet.Select(c => c.Index));

            foreach (var constraint in constraints)
            {
                if (activeLookup.Contains(constraint.Index))
                {
                    continue;
                }

                var numerator = constraint.RightHandSide - constraint.Coefficients.DotProduct(point);
                var denominator = constraint.Coefficients.DotProduct(direction);

                if (denominator > tolerance)
                {
                    var candidate = numerator / denominator;
                    if (candidate >= -tolerance && candidate < step)
                    {
                        step = Math.Max(candidate, 0.0);
                        blockingConstraint = constraint;
                    }
                }
            }

            return (step, blockingConstraint);
        }

        private static double EvaluateObjective(Matrix<double> q, Vector<double> c, Vector<double> x)
        {
            var quadratic = x.DotProduct(q * x) * 0.5;
            var linear = c.DotProduct(x);
            return quadratic + linear;
        }

        private static double[,] ToArray(Matrix<double> matrix)
        {
            var result = new double[matrix.RowCount, matrix.ColumnCount];
            for (var i = 0; i < matrix.RowCount; i++)
            {
                for (var j = 0; j < matrix.ColumnCount; j++)
                {
                    result[i, j] = matrix[i, j];
                }
            }

            return result;
        }

        private enum ConstraintSource
        {
            Equality,
            Inequality,
            LowerBound,
            UpperBound
        }

        private sealed class ConstraintBinding
        {
            public int Index { get; set; }

            public int OriginalIndex { get; set; }

            public ConstraintSource Source { get; set; }

            public bool IsEquality { get; set; }

            public Vector<double> Coefficients { get; set; }

            public double RightHandSide { get; set; }
        }
    }
}
