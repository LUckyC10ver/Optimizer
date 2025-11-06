using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;
using Optimizer.Core.QuadraticProgramming;

namespace Optimizer.Core.NonlinearProgramming
{
    public class SequentialQuadraticProgrammingSolver
    {
        private readonly QuadraticProgrammingSolver _quadraticSolver = new QuadraticProgrammingSolver();

        public Solution Solve(
            NonlinearProblem problem,
            SqpOptions options,
            out SqpInfo info,
            Matrix<double> initialHessian = null,
            out Vector<double> multipliers,
            out IReadOnlyList<int> activeSet)
        {
            if (problem == null)
            {
                throw new OptimizationException("Nonlinear problem cannot be null.");
            }

            options ??= new SqpOptions();
            info = new SqpInfo();
            var stopwatch = Stopwatch.StartNew();

            var dimension = problem.VariableCount;
            if (dimension <= 0)
            {
                if (problem.InitialGuess == null)
                {
                    throw new OptimizationException("The problem definition must supply either a variable count or an initial guess.");
                }

                dimension = problem.InitialGuess.Count;
            }

            var x = problem.InitialGuess?.Clone() ?? Vector<double>.Build.Dense(dimension);
            var constraintTolerance = ResolveConstraintTolerance(options);
            var maxIterations = options.MaxIterations > 0 ? options.MaxIterations : 100;
            var maxFunctionEvals = options.MaxFunctionEvaluations > 0 ? options.MaxFunctionEvaluations : maxIterations * (dimension + 1);

            var hessian = initialHessian != null && initialHessian.RowCount == dimension && initialHessian.ColumnCount == dimension
                ? initialHessian.Clone()
                : Matrix<double>.Build.DenseIdentity(dimension);

            var (objectiveValue, gradient) = EvaluateObjective(problem, x, true, options, info);
            var constraintData = EvaluateConstraints(problem, x, true, options, info);

            var penalty = Math.Max(options.InitialPenalty, 1.0);
            var solution = new Solution
            {
                Status = SolverResultStatus.Unknown,
                OptimalPoint = x.Clone(),
                OptimalValue = objectiveValue
            };

            var qpOptions = new SolverOptions
            {
                Tolerance = Math.Max(options.ObjectiveTolerance, 1e-10),
                ConstraintTolerance = constraintTolerance,
                MaxIterations = options.MaxQuadraticIterations > 0 ? options.MaxQuadraticIterations : 200,
                Verbose = options.Display > 2
            };

            var lowerBounds = problem.LowerBounds;
            var upperBounds = problem.UpperBounds;

            multipliers = Vector<double>.Build.Dense(constraintData.TotalConstraintCount);
            activeSet = Array.Empty<int>();
            for (info.SqpCount = 0; info.SqpCount < maxIterations; info.SqpCount++)
            {
                info.ObjectiveValue = objectiveValue;

                var qpProblem = BuildQuadraticSubProblem(
                    x,
                    hessian,
                    gradient,
                    constraintData,
                    lowerBounds,
                    upperBounds,
                    options);

                Solution qpSolution;
                try
                {
                    qpSolution = _quadraticSolver.Solve(qpProblem, qpOptions);
                }
                catch (Exception ex)
                {
                    info.QpStatus = ex.Message;
                    solution.Status = SolverResultStatus.Error;
                    solution.Message = $"Quadratic sub-problem failed: {ex.Message}";
                    solution.OptimalPoint = x.Clone();
                    solution.OptimalValue = objectiveValue;
                    solution.SolveTime = stopwatch.Elapsed;
                    return solution;
                }

                info.QpStatus = qpSolution.Message;

                if (qpSolution.Status != SolverResultStatus.Optimal)
                {
                    solution.Status = qpSolution.Status == SolverResultStatus.IterationLimit
                        ? SolverResultStatus.IterationLimit
                        : SolverResultStatus.Error;
                    solution.Message = qpSolution.Message;
                    solution.OptimalPoint = x.Clone();
                    solution.OptimalValue = objectiveValue;
                    solution.SolveTime = stopwatch.Elapsed;
                    return solution;
                }

                var step = qpSolution.OptimalPoint;
                var stepNorm = step.L2Norm();
                if (stepNorm <= options.ArgumentTolerance)
                {
                    if (constraintData.MaximumViolation <= constraintTolerance)
                    {
                        solution.Status = SolverResultStatus.Optimal;
                    }
                    else
                    {
                        solution.Status = SolverResultStatus.IterationLimit;
                        solution.Message = "Terminated due to small search direction while constraints are still violated.";
                    }

                    solution.OptimalPoint = x.Clone();
                    solution.OptimalValue = objectiveValue;
                    solution.SolveTime = stopwatch.Elapsed;
                    multipliers = ExtractMultipliers(qpSolution.Message, constraintData.TotalConstraintCount);
                    activeSet = ExtractActiveSet(qpSolution.Message);
                    return solution;
                }

                var directionalDerivative = gradient.DotProduct(step);
                var currentMerit = Merit(objectiveValue, constraintData.EqualityValues, constraintData.InequalityValues, penalty);

                var lineSearchResult = PerformLineSearch(
                    problem,
                    x,
                    step,
                    penalty,
                    options,
                    info,
                    lowerBounds,
                    upperBounds,
                    gradient,
                    objectiveValue,
                    constraintData,
                    directionalDerivative,
                    currentMerit);

                if (!lineSearchResult.Success)
                {
                    solution.Status = SolverResultStatus.IterationLimit;
                    solution.Message = "Line search failed to locate a descent step.";
                    solution.OptimalPoint = x.Clone();
                    solution.OptimalValue = objectiveValue;
                    solution.SolveTime = stopwatch.Elapsed;
                    return solution;
                }

                var displacement = lineSearchResult.Point - x;
                var gradientChange = lineSearchResult.Gradient - gradient;
                UpdateHessian(hessian, displacement, gradientChange, options);

                x = lineSearchResult.Point;
                objectiveValue = lineSearchResult.Objective;
                gradient = lineSearchResult.Gradient;
                constraintData = lineSearchResult.Constraints;
                info.StepLength = lineSearchResult.StepLength;

                penalty = UpdatePenalty(penalty, constraintData.MaximumViolation);

                if (info.FunctionCount >= maxFunctionEvals)
                {
                    solution.Status = SolverResultStatus.IterationLimit;
                    solution.Message = "Maximum number of function evaluations reached.";
                    solution.OptimalPoint = x.Clone();
                    solution.OptimalValue = objectiveValue;
                    solution.SolveTime = stopwatch.Elapsed;
                    return solution;
                }

                if (gradient.L2Norm() <= options.ObjectiveTolerance && constraintData.MaximumViolation <= constraintTolerance)
                {
                    solution.Status = SolverResultStatus.Optimal;
                    solution.OptimalPoint = x.Clone();
                    solution.OptimalValue = objectiveValue;
                    solution.SolveTime = stopwatch.Elapsed;
                    multipliers = ExtractMultipliers(info.QpStatus, constraintData.TotalConstraintCount);
                    activeSet = ExtractActiveSet(info.QpStatus);
                    return solution;
                }
            }

            solution.Status = SolverResultStatus.IterationLimit;
            solution.Message = "Maximum number of SQP iterations reached.";
            solution.OptimalPoint = x.Clone();
            solution.OptimalValue = objectiveValue;
            solution.SolveTime = stopwatch.Elapsed;
            multipliers = ExtractMultipliers(info.QpStatus, constraintData.TotalConstraintCount);
            activeSet = ExtractActiveSet(info.QpStatus);
            return solution;
        }

        public Solution Solve(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IEnumerable<NonlinearConstraint> constraints,
            Vector<double> initialGuess,
            SqpOptions options,
            out SqpInfo info)
        {
            var constraintList = constraints != null
                ? new List<NonlinearConstraint>(constraints)
                : new List<NonlinearConstraint>();

            var problem = new NonlinearProblem(
                objective,
                gradient,
                constraintList,
                initialGuess,
                initialGuess?.Count ?? 0);

            return Solve(problem, options, out info, null, out _, out _);
        }

        private static double ResolveConstraintTolerance(SqpOptions options)
        {
            if (options.ConstraintTolerance > 0)
            {
                return options.ConstraintTolerance;
            }

            if (options.ConstraintToleranceOverride > 0)
            {
                return options.ConstraintToleranceOverride;
            }

            return 1e-6;
        }

        private static (double Value, Vector<double> Gradient) EvaluateObjective(
            NonlinearProblem problem,
            Vector<double> point,
            bool computeGradient,
            SqpOptions options,
            SqpInfo info)
        {
            if (problem.Objective == null)
            {
                throw new OptimizationException("The objective function has not been provided.");
            }

            var value = problem.Objective(point);
            info.FunctionCount++;

            Vector<double> gradient;
            if (computeGradient)
            {
                if (problem.Gradient != null)
                {
                    gradient = problem.Gradient(point);
                    info.GradientCount++;
                }
                else
                {
                    gradient = ApproximateGradient(point, problem.Objective, options);
                    info.GradientCount++;
                }
            }
            else
            {
                gradient = Vector<double>.Build.Dense(point.Count);
            }

            return (value, gradient);
        }

        private static ConstraintEvaluation EvaluateConstraints(
            NonlinearProblem problem,
            Vector<double> point,
            bool computeJacobian,
            SqpOptions options,
            SqpInfo info)
        {
            var equalityValues = new List<double>();
            var inequalityValues = new List<double>();
            var equalityRows = new List<Vector<double>>();
            var inequalityRows = new List<Vector<double>>();

            if (problem.LinearEqualityMatrix != null && problem.LinearEqualityVector != null)
            {
                for (var i = 0; i < problem.LinearEqualityMatrix.RowCount; i++)
                {
                    var row = problem.LinearEqualityMatrix.Row(i);
                    equalityValues.Add(row.DotProduct(point) - problem.LinearEqualityVector[i]);
                    if (computeJacobian)
                    {
                        equalityRows.Add(row);
                    }
                }
            }

            if (problem.LinearInequalityMatrix != null && problem.LinearInequalityVector != null)
            {
                for (var i = 0; i < problem.LinearInequalityMatrix.RowCount; i++)
                {
                    var row = problem.LinearInequalityMatrix.Row(i);
                    inequalityValues.Add(row.DotProduct(point) - problem.LinearInequalityVector[i]);
                    if (computeJacobian)
                    {
                        inequalityRows.Add(row);
                    }
                }
            }

            foreach (var constraint in problem.Constraints)
            {
                if (constraint.Function == null)
                {
                    throw new OptimizationException("Encountered nonlinear constraint without a function delegate.");
                }

                var value = constraint.Function(point);
                info.FunctionCount++;

                var gradient = computeJacobian
                    ? EvaluateConstraintGradient(constraint, point, options, info)
                    : Vector<double>.Build.Dense(point.Count);

                switch (constraint.Type)
                {
                    case ConstraintType.Equal:
                        equalityValues.Add(value);
                        if (computeJacobian)
                        {
                            equalityRows.Add(gradient);
                        }

                        break;
                    case ConstraintType.LessEqual:
                        inequalityValues.Add(value);
                        if (computeJacobian)
                        {
                            inequalityRows.Add(gradient);
                        }

                        break;
                    case ConstraintType.GreaterEqual:
                        inequalityValues.Add(-value);
                        if (computeJacobian)
                        {
                            inequalityRows.Add(-gradient);
                        }

                        break;
                    default:
                        throw new OptimizationException("Unsupported constraint type encountered in SQP solver.");
                }
            }

            var eqVector = equalityValues.Count > 0
                ? Vector<double>.Build.DenseOfEnumerable(equalityValues)
                : Vector<double>.Build.Dense(0);

            var ineqVector = inequalityValues.Count > 0
                ? Vector<double>.Build.DenseOfEnumerable(inequalityValues)
                : Vector<double>.Build.Dense(0);

            var eqMatrix = equalityRows.Count > 0
                ? Matrix<double>.Build.DenseOfRowVectors(equalityRows)
                : Matrix<double>.Build.Dense(0, point.Count);

            var ineqMatrix = inequalityRows.Count > 0
                ? Matrix<double>.Build.DenseOfRowVectors(inequalityRows)
                : Matrix<double>.Build.Dense(0, point.Count);

            var maxEq = eqVector.Count > 0 ? eqVector.AbsoluteMaximum() : 0.0;
            var maxIneq = ineqVector.Count > 0 ? ineqVector.Select(v => Math.Max(0.0, v)).DefaultIfEmpty(0.0).Max() : 0.0;

            return new ConstraintEvaluation(eqVector, ineqVector, eqMatrix, ineqMatrix, Math.Max(maxEq, maxIneq));
        }

        private static Vector<double> EvaluateConstraintGradient(NonlinearConstraint constraint, Vector<double> point, SqpOptions options, SqpInfo info)
        {
            if (constraint.Gradient != null)
            {
                info.GradientCount++;
                return constraint.Gradient(point);
            }

            info.GradientCount++;
            return ApproximateGradient(point, constraint.Function, options);
        }

        private static Vector<double> ApproximateGradient(Vector<double> point, Func<Vector<double>, double> func, SqpOptions options)
        {
            var n = point.Count;
            var gradient = Vector<double>.Build.Dense(n);
            var baseValue = func(point);

            var step = Vector<double>.Build.Dense(n);
            for (var i = 0; i < n; i++)
            {
                var magnitude = Math.Max(Math.Abs(point[i]), 1.0);
                var delta = magnitude * Math.Sqrt(options.MachineEpsilon);
                delta = Math.Min(Math.Max(delta, options.FiniteDifferenceMinChange), options.FiniteDifferenceMaxChange);
                step[i] = delta;
            }

            for (var i = 0; i < n; i++)
            {
                var backup = point[i];
                point[i] = backup + step[i];
                var forward = func(point);
                point[i] = backup - step[i];
                var backward = func(point);
                point[i] = backup;
                gradient[i] = (forward - backward) / (2.0 * step[i]);
            }

            return gradient;
        }

        private static QuadraticProblem BuildQuadraticSubProblem(
            Vector<double> point,
            Matrix<double> hessian,
            Vector<double> gradient,
            ConstraintEvaluation constraints,
            Vector<double> lowerBounds,
            Vector<double> upperBounds,
            SqpOptions options)
        {
            var equalityVector = constraints.EqualityValues.Count > 0
                ? -constraints.EqualityValues
                : Vector<double>.Build.Dense(0);

            var inequalityVector = constraints.InequalityValues.Count > 0
                ? -constraints.InequalityValues
                : Vector<double>.Build.Dense(0);

            var qpLower = lowerBounds != null
                ? lowerBounds - point
                : null;

            var qpUpper = upperBounds != null
                ? upperBounds - point
                : null;

            return new QuadraticProblem(
                hessian.Clone(),
                gradient.Clone(),
                constraints.InequalityJacobian.RowCount > 0 ? constraints.InequalityJacobian.Clone() : null,
                inequalityVector.Count > 0 ? inequalityVector.Clone() : null,
                true,
                constraints.EqualityJacobian.RowCount > 0 ? constraints.EqualityJacobian.Clone() : null,
                equalityVector.Count > 0 ? equalityVector.Clone() : null,
                qpLower,
                qpUpper,
                Vector<double>.Build.Dense(point.Count));
        }

        private static double Merit(double objective, Vector<double> equality, Vector<double> inequality, double penalty)
        {
            var eqPenalty = equality.Count > 0 ? equality.L1Norm() : 0.0;
            var ineqPenalty = inequality.Count > 0
                ? inequality.Select(v => Math.Max(0.0, v)).Sum()
                : 0.0;
            return objective + penalty * (eqPenalty + ineqPenalty);
        }

        private static void UpdateHessian(Matrix<double> hessian, Vector<double> s, Vector<double> y, SqpOptions options)
        {
            var sy = s.DotProduct(y);
            if (sy <= 0)
            {
                ResetToIdentity(hessian);
                return;
            }

            var Bs = hessian * s;
            var sBs = s.DotProduct(Bs);
            if (sBs <= options.MachineEpsilon)
            {
                ResetToIdentity(hessian);
                return;
            }

            var yyT = y.OuterProduct(y) / sy;
            var BsOuter = Bs.OuterProduct(Bs) / sBs;
            var updated = hessian - BsOuter + yyT;
            for (var i = 0; i < hessian.RowCount; i++)
            {
                for (var j = 0; j < hessian.ColumnCount; j++)
                {
                    hessian[i, j] = updated[i, j];
                }
            }
        }

        private static void ResetToIdentity(Matrix<double> matrix)
        {
            for (var i = 0; i < matrix.RowCount; i++)
            {
                for (var j = 0; j < matrix.ColumnCount; j++)
                {
                    matrix[i, j] = i == j ? 1.0 : 0.0;
                }
            }
        }

        private static Vector<double> ProjectOntoBounds(Vector<double> candidate, Vector<double> lower, Vector<double> upper)
        {
            if (lower == null && upper == null)
            {
                return candidate;
            }

            var projected = candidate.Clone();
            for (var i = 0; i < projected.Count; i++)
            {
                var value = projected[i];
                if (lower != null)
                {
                    value = Math.Max(value, lower[i]);
                }

                if (upper != null)
                {
                    value = Math.Min(value, upper[i]);
                }

                projected[i] = value;
            }

            return projected;
        }

        private LineSearchResult PerformLineSearch(
            NonlinearProblem problem,
            Vector<double> currentPoint,
            Vector<double> step,
            double penalty,
            SqpOptions options,
            SqpInfo info,
            Vector<double> lowerBounds,
            Vector<double> upperBounds,
            Vector<double> currentGradient,
            double currentObjective,
            ConstraintEvaluation currentConstraints,
            double directionalDerivative,
            double currentMerit)
        {
            var maxBacktrackingSteps = options.MaxLineSearchSteps > 0 ? options.MaxLineSearchSteps : 20;

            var cache = new Dictionary<double, LineSearchEvaluation>
            {
                [0.0] = new LineSearchEvaluation(
                    currentPoint.Clone(),
                    currentObjective,
                    currentGradient.Clone(),
                    currentConstraints,
                    currentMerit)
            };

            LineSearchEvaluation Evaluate(double stepLength, bool computeGradient)
            {
                stepLength = Math.Max(0.0, stepLength);

                if (cache.TryGetValue(stepLength, out var stored) && (!computeGradient || stored.HasGradient))
                {
                    return stored;
                }

                if (stepLength <= 0.0)
                {
                    return cache[0.0];
                }

                var candidate = ProjectOntoBounds(currentPoint + stepLength * step, lowerBounds, upperBounds);
                var (value, gradientCandidate) = EvaluateObjective(problem, candidate, computeGradient, options, info);
                var constraintsCandidate = EvaluateConstraints(problem, candidate, computeGradient, options, info);
                var meritCandidate = Merit(value, constraintsCandidate.EqualityValues, constraintsCandidate.InequalityValues, penalty);

                var evaluation = new LineSearchEvaluation(
                    candidate,
                    value,
                    computeGradient ? gradientCandidate.Clone() : null,
                    constraintsCandidate,
                    meritCandidate);

                cache[stepLength] = evaluation;
                return evaluation;
            }

            double MeritAt(double stepLength)
            {
                var evaluation = Evaluate(stepLength, computeGradient: false);
                return double.IsNaN(evaluation.Merit) ? double.PositiveInfinity : evaluation.Merit;
            }

            LineSearchResult Backtracking()
            {
                var stepLength = 1.0;
                for (var attempt = 0; attempt < maxBacktrackingSteps; attempt++)
                {
                    var evaluation = Evaluate(stepLength, computeGradient: true);
                    if (!double.IsNaN(evaluation.Merit) &&
                        evaluation.Merit <= currentMerit + options.ArmijoFactor * stepLength * directionalDerivative)
                    {
                        return new LineSearchResult(true, stepLength, evaluation.Point, evaluation.Objective, evaluation.Gradient, evaluation.Constraints, evaluation.Merit);
                    }

                    stepLength *= options.LineSearchShrink;
                    if (stepLength < options.MinimumStepLength)
                    {
                        break;
                    }
                }

                return LineSearchResult.Failure;
            }

            LineSearchResult Brent()
            {
                try
                {
                    var initialB = Math.Min(0.5, 1.0);
                    var (a, b, c, _, _, _) = LineSearchTools.BracketMinimum(0.0, initialB, 1.0, MeritAt);
                    var tolerance = Math.Max(options.ArgumentTolerance, 1e-4);
                    (double value, double position) searchResult = options.LineSearch == 1
                        ? LineSearchTools.GoldenSectionSearch(a, b, c, MeritAt, tolerance)
                        : LineSearchTools.BrentSearch(a, b, c, MeritAt, tolerance, Math.Max(maxBacktrackingSteps, 50));

                    var stepLength = Math.Max(options.MinimumStepLength, Math.Min(1.0, searchResult.position));
                    if (stepLength <= options.MinimumStepLength)
                    {
                        return LineSearchResult.Failure;
                    }

                    var evaluation = Evaluate(stepLength, computeGradient: true);
                    if (double.IsNaN(evaluation.Merit) || evaluation.Merit >= currentMerit)
                    {
                        return LineSearchResult.Failure;
                    }

                    return new LineSearchResult(true, stepLength, evaluation.Point, evaluation.Objective, evaluation.Gradient, evaluation.Constraints, evaluation.Merit);
                }
                catch
                {
                    return LineSearchResult.Failure;
                }
            }

            if (options.LineSearch <= 0)
            {
                return Backtracking();
            }

            var result = Brent();
            if (!result.Success)
            {
                result = Backtracking();
            }

            return result;
        }

        private static double UpdatePenalty(double currentPenalty, double violation)
        {
            if (violation <= 1e-8)
            {
                return Math.Max(currentPenalty * 0.8, 1.0);
            }

            return Math.Min(currentPenalty * 1.2 + violation, currentPenalty * 5.0);
        }

        private static Vector<double> ExtractMultipliers(string message, int length)
        {
            if (string.IsNullOrWhiteSpace(message) || length <= 0)
            {
                return Vector<double>.Build.Dense(length);
            }

            var parts = message.Split(new[] { ' ', ',', ';' }, StringSplitOptions.RemoveEmptyEntries);
            var values = new List<double>();
            foreach (var part in parts)
            {
                if (double.TryParse(part, out var value))
                {
                    values.Add(value);
                }
            }

            if (values.Count == length)
            {
                return Vector<double>.Build.DenseOfEnumerable(values);
            }

            return Vector<double>.Build.Dense(length);
        }

        private static IReadOnlyList<int> ExtractActiveSet(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return Array.Empty<int>();
            }

            var indices = new List<int>();
            foreach (var token in message.Split(new[] { ' ', ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (int.TryParse(token, out var value))
                {
                    indices.Add(value);
                }
            }

            return indices;
        }

        private readonly struct LineSearchResult
        {
            public LineSearchResult(
                bool success,
                double stepLength,
                Vector<double> point,
                double objective,
                Vector<double> gradient,
                ConstraintEvaluation constraints,
                double merit)
            {
                Success = success;
                StepLength = stepLength;
                Point = point;
                Objective = objective;
                Gradient = gradient;
                Constraints = constraints;
                Merit = merit;
            }

            public bool Success { get; }

            public double StepLength { get; }

            public Vector<double> Point { get; }

            public double Objective { get; }

            public Vector<double> Gradient { get; }

            public ConstraintEvaluation Constraints { get; }

            public double Merit { get; }

            public static LineSearchResult Failure => new LineSearchResult(false, 0.0, null, double.NaN, null, default, double.PositiveInfinity);
        }

        private readonly struct LineSearchEvaluation
        {
            public LineSearchEvaluation(
                Vector<double> point,
                double objective,
                Vector<double> gradient,
                ConstraintEvaluation constraints,
                double merit)
            {
                Point = point;
                Objective = objective;
                Gradient = gradient;
                Constraints = constraints;
                Merit = merit;
            }

            public Vector<double> Point { get; }

            public double Objective { get; }

            public Vector<double> Gradient { get; }

            public ConstraintEvaluation Constraints { get; }

            public double Merit { get; }

            public bool HasGradient => Gradient != null;
        }

        private readonly struct ConstraintEvaluation
        {
            public ConstraintEvaluation(
                Vector<double> equalityValues,
                Vector<double> inequalityValues,
                Matrix<double> equalityJacobian,
                Matrix<double> inequalityJacobian,
                double maximumViolation)
            {
                EqualityValues = equalityValues;
                InequalityValues = inequalityValues;
                EqualityJacobian = equalityJacobian;
                InequalityJacobian = inequalityJacobian;
                MaximumViolation = maximumViolation;
            }

            public Vector<double> EqualityValues { get; }

            public Vector<double> InequalityValues { get; }

            public Matrix<double> EqualityJacobian { get; }

            public Matrix<double> InequalityJacobian { get; }

            public double MaximumViolation { get; }

            public int TotalConstraintCount => EqualityValues.Count + InequalityValues.Count;
        }
    }
}
