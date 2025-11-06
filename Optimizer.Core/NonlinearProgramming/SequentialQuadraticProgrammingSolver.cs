using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Simplified SQP-style optimiser that uses a penalty approach but preserves the classic interface.
    /// </summary>
    public sealed class SequentialQuadraticProgrammingSolver
    {
        public Vector<double> Solve(
            NonlinearProblem problem,
            Func<Vector<double>, ConstraintEvaluation> constraintEvaluator,
            SqpOptions options,
            SqpInfo info,
            out Vector<double> multipliers,
            out List<int> activeSet)
        {
            if (problem == null)
            {
                throw new ArgumentNullException(nameof(problem));
            }

            options ??= new SqpOptions();
            info ??= new SqpInfo();

            var maxIterations = options.MaxIterations > 0
                ? options.MaxIterations
                : options.MaxFunEvals > 0 ? options.MaxFunEvals : 500;
            var tolerance = options.Tolerance > 0
                ? options.Tolerance
                : Math.Min(Math.Min(options.TolArg, options.TolCon), options.TolObj);
            var penalty = options.PenaltyWeight;
            var step = options.StepSize;
            var diffStep = options.FiniteDifferenceStep > 0
                ? options.FiniteDifferenceStep
                : options.DiffMinChange;

            var x = problem.InitialGuess.Clone();
            var dimension = x.Count;
            var gradient = Vector<double>.Build.Dense(dimension);
            var best = x.Clone();

            double bestObjective = double.PositiveInfinity;
            double bestViolation = double.PositiveInfinity;
            double bestRawObjective = double.PositiveInfinity;
            double lastRawObjective = double.PositiveInfinity;
            double lastViolation = double.PositiveInfinity;

            var equalityCount = problem.EqualityMatrix?.RowCount ?? 0;
            var inequalityCount = problem.InequalityMatrix?.RowCount ?? 0;
            multipliers = Vector<double>.Build.Dense(equalityCount + inequalityCount);
            activeSet = new List<int>();

            for (var iteration = 0; iteration < maxIterations; iteration++)
            {
                info.SqpCount = iteration + 1;
                info.FunCount++;

                var composite = EvaluateComposite(problem, constraintEvaluator, x, penalty, out var rawObjective, out var violation);
                lastRawObjective = rawObjective;
                lastViolation = violation;

                if (composite < bestObjective)
                {
                    bestObjective = composite;
                    bestViolation = violation;
                    bestRawObjective = rawObjective;
                    best.SetSubVector(0, dimension, x);
                }

                if (violation < tolerance)
                {
                    info.How = "Feasible";
                }

                ApproximateGradient(problem, constraintEvaluator, x, penalty, diffStep, gradient, out var gradNorm);
                info.GradCount++;
                info.GradientNorm = gradNorm;

                if (gradNorm < tolerance && violation < options.TolCon)
                {
                    ReportProgress(options, info.SqpCount, rawObjective, violation, info);
                    break;
                }

                x -= gradient * step;
                ProjectBounds(problem, x);
                EnforceLinearEqualities(problem, x);

                ReportProgress(options, info.SqpCount, rawObjective, violation, info);
            }

            info.ObjectiveValue = bestRawObjective;
            info.ConstraintViolation = bestViolation;
            info.StepLength = step;
            info.Status = bestViolation <= options.TolCon && info.GradientNorm <= options.TolObj ? "Converged" : "Finished";

            ExtractActiveSet(problem, constraintEvaluator, best, options, activeSet);
            multipliers = Vector<double>.Build.Dense(activeSet.Count);

            ReportProgress(options, info.SqpCount, lastRawObjective, lastViolation, info);

            return best;
        }

        private static void ReportProgress(
            SqpOptions options,
            int iteration,
            double rawObjective,
            double violation,
            SqpInfo info)
        {
            if (options == null)
            {
                return;
            }

            options.ProgressCallback?.Invoke(iteration, rawObjective, violation, info);
            options.ProgressReporter?.Invoke(info);
        }

        private static void ProjectBounds(NonlinearProblem problem, Vector<double> x)
        {
            if (problem.LowerBounds != null)
            {
                for (var i = 0; i < x.Count && i < problem.LowerBounds.Count; i++)
                {
                    x[i] = Math.Max(x[i], problem.LowerBounds[i]);
                }
            }

            if (problem.UpperBounds != null)
            {
                for (var i = 0; i < x.Count && i < problem.UpperBounds.Count; i++)
                {
                    x[i] = Math.Min(x[i], problem.UpperBounds[i]);
                }
            }
        }

        private static void EnforceLinearEqualities(NonlinearProblem problem, Vector<double> x)
        {
            if (problem.EqualityMatrix == null || problem.EqualityVector == null)
            {
                return;
            }

            var residual = problem.EqualityMatrix * x - problem.EqualityVector;
            if (residual.L2Norm() < 1e-12)
            {
                return;
            }

            var correction = problem.EqualityMatrix.TransposeThisAndMultiply(residual);
            var scale = correction.L2Norm();
            if (scale > 0)
            {
                correction /= scale * scale;
                x -= correction;
            }
        }

        private static void ApproximateGradient(
            NonlinearProblem problem,
            Func<Vector<double>, ConstraintEvaluation> constraintEvaluator,
            Vector<double> point,
            double penalty,
            double diffStep,
            Vector<double> gradient,
            out double norm)
        {
            var dimension = point.Count;
            var baseValue = EvaluateComposite(problem, constraintEvaluator, point, penalty, out _, out _);
            norm = 0.0;

            for (var i = 0; i < dimension; i++)
            {
                var saved = point[i];
                point[i] = saved + diffStep;
                var forward = EvaluateComposite(problem, constraintEvaluator, point, penalty, out _, out _);
                point[i] = saved - diffStep;
                var backward = EvaluateComposite(problem, constraintEvaluator, point, penalty, out _, out _);
                point[i] = saved;

                var derivative = (forward - backward) / (2.0 * diffStep);
                gradient[i] = derivative;
                norm += derivative * derivative;
            }

            norm = Math.Sqrt(norm);
        }

        private static double EvaluateComposite(
            NonlinearProblem problem,
            Func<Vector<double>, ConstraintEvaluation> constraintEvaluator,
            Vector<double> x,
            double penalty,
            out double rawObjective,
            out double violation)
        {
            rawObjective = problem.Objective(x);
            violation = 0.0;
            var penaltyValue = 0.0;

            if (problem.EqualityMatrix != null && problem.EqualityVector != null)
            {
                var residual = problem.EqualityMatrix * x - problem.EqualityVector;
                penaltyValue += residual.DotProduct(residual);
                violation = Math.Max(violation, residual.L2Norm());
            }

            if (problem.InequalityMatrix != null && problem.InequalityVector != null)
            {
                var residual = problem.InequalityMatrix * x - problem.InequalityVector;
                for (var i = 0; i < residual.Count; i++)
                {
                    if (residual[i] > 0)
                    {
                        penaltyValue += residual[i] * residual[i];
                        violation = Math.Max(violation, residual[i]);
                    }
                }
            }

            foreach (var constraint in problem.Constraints)
            {
                var value = constraint.Function(x);
                switch (constraint.Kind)
                {
                    case ConstraintKind.Equality:
                        penaltyValue += value * value;
                        violation = Math.Max(violation, Math.Abs(value));
                        break;
                    case ConstraintKind.LessOrEqual:
                        if (value > 0)
                        {
                            penaltyValue += value * value;
                            violation = Math.Max(violation, value);
                        }
                        break;
                    case ConstraintKind.GreaterOrEqual:
                        if (value < 0)
                        {
                            var deficit = -value;
                            penaltyValue += deficit * deficit;
                            violation = Math.Max(violation, deficit);
                        }
                        break;
                }
            }

            if (constraintEvaluator != null)
            {
                var evaluation = constraintEvaluator(x) ?? ConstraintEvaluation.Empty;
                if (evaluation.Values != null)
                {
                    for (var i = 0; i < evaluation.Values.Count; i++)
                    {
                        var value = evaluation.Values[i];
                        if (i < evaluation.EqualityCount)
                        {
                            penaltyValue += value * value;
                            violation = Math.Max(violation, Math.Abs(value));
                        }
                        else if (value > 0)
                        {
                            penaltyValue += value * value;
                            violation = Math.Max(violation, value);
                        }
                    }
                }
            }

            return rawObjective + penalty * penaltyValue;
        }

        private static void ExtractActiveSet(
            NonlinearProblem problem,
            Func<Vector<double>, ConstraintEvaluation> constraintEvaluator,
            Vector<double> x,
            SqpOptions options,
            List<int> activeSet)
        {
            activeSet.Clear();
            var index = 0;

            if (problem.EqualityMatrix != null)
            {
                for (var i = 0; i < problem.EqualityMatrix.RowCount; i++)
                {
                    activeSet.Add(index++);
                }
            }

            if (problem.InequalityMatrix != null && problem.InequalityVector != null)
            {
                var residual = problem.InequalityMatrix * x - problem.InequalityVector;
                for (var i = 0; i < residual.Count; i++)
                {
                    if (Math.Abs(residual[i]) <= options.TolCon)
                    {
                        activeSet.Add(index + i);
                    }
                }
                index += residual.Count;
            }

            if (problem.Constraints != null)
            {
                foreach (var constraint in problem.Constraints)
                {
                    var value = constraint.Function(x);
                    var isActive = constraint.Kind switch
                    {
                        ConstraintKind.Equality => true,
                        ConstraintKind.LessOrEqual => value <= options.TolCon,
                        ConstraintKind.GreaterOrEqual => value >= -options.TolCon,
                        _ => false
                    };

                    if (isActive)
                    {
                        activeSet.Add(index);
                    }

                    index++;
                }
            }

            if (constraintEvaluator != null)
            {
                var evaluation = constraintEvaluator(x) ?? ConstraintEvaluation.Empty;
                if (evaluation.Values != null)
                {
                    for (var i = 0; i < evaluation.Values.Count; i++)
                    {
                        var value = evaluation.Values[i];
                        var isActive = i < evaluation.EqualityCount
                            ? true
                            : value <= options.TolCon;

                        if (isActive)
                        {
                            activeSet.Add(index + i);
                        }
                    }
                }
            }
        }
    }
}
