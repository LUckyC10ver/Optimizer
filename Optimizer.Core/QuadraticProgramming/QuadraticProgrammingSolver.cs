using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
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

            if (options == null)
            {
                options = new SolverOptions();
            }
            var tolerance = options.Tolerance <= 0 ? 1e-8 : options.Tolerance;
            var maxIterations = Math.Max(1, options.MaxIterations);
            var penaltyWeight = 10.0;
            var stepSize = EstimateInitialStep(problem, penaltyWeight);
            var minStepSize = 1e-12;

            var writer = ResolveWriter(options);
            writer?.WriteLine("[quadprog] Starting optimisation");
            writer?.WriteLine($"[quadprog] Settings: tolerance={FormatDouble(tolerance)}, maxIterations={maxIterations}, stepSize={FormatDouble(stepSize)}");

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

            writer?.WriteLine($"[quadprog] Initial iterate: {FormatVector(x)}");

            var gradient = Vector<double>.Build.Dense(dimension);
            var candidate = Vector<double>.Build.Dense(dimension);
            var watch = Stopwatch.StartNew();
            var iterations = 0;
            var status = SolverResultStatus.IterationLimit;

            var currentPenaltyObjective = EvaluatePenalty(problem, x, penaltyWeight);

            for (; iterations < maxIterations; iterations++)
            {
                if (HasTimeLimitExpired(options, watch.Elapsed))
                {
                    status = SolverResultStatus.TimeLimit;
                    writer?.WriteLine("[quadprog] Time limit reached, stopping optimisation.");
                    break;
                }

                ComputeGradient(problem, x, penaltyWeight, gradient, out var violation);
                var objective = EvaluateObjective(problem, x);
                if (!problem.IsMinimisation)
                {
                    objective = -objective;
                }

                var gradientNorm = gradient.L2Norm();
                EvaluateResiduals(problem, x, out var equalityNorm, out var inequalityViolation);

                var iterationIndex = iterations + 1;
                writer?.WriteLine(
                    $"[quadprog] iter {iterationIndex:D4}: obj={FormatDouble(objective)}, grad={FormatDouble(gradientNorm)}, eqNorm={FormatDouble(equalityNorm)}, ineqPos={FormatDouble(inequalityViolation)}, maxViol={FormatDouble(violation)}");

                if (violation < tolerance && gradientNorm < tolerance)
                {
                    status = SolverResultStatus.Optimal;
                    writer?.WriteLine("[quadprog] Converged based on tolerance criteria.");
                    iterations = iterationIndex;
                    break;
                }

                var stepTaken = PerformBacktrackingStep(problem, x, gradient, penaltyWeight, stepSize, minStepSize, candidate, writer, tolerance, out var acceptedStepNorm, ref currentPenaltyObjective);

                if (!stepTaken)
                {
                    status = SolverResultStatus.IterationLimit;
                    writer?.WriteLine("[quadprog] Step could not be taken (minimum step size reached).");
                    iterations = iterationIndex;
                    break;
                }

                writer?.WriteLine($"[quadprog]          step={FormatDouble(acceptedStepNorm)} -> new x={FormatVector(x)}");

                if (iterationIndex % 25 == 0 && violation > tolerance)
                {
                    penaltyWeight *= 2.0;
                    writer?.WriteLine($"[quadprog] Increasing penalty weight to {FormatDouble(penaltyWeight)}");
                }
            }

            watch.Stop();

            var finalObjective = EvaluateObjective(problem, x);
            if (!problem.IsMinimisation)
            {
                finalObjective = -finalObjective;
            }

            writer?.WriteLine($"[quadprog] Finished with status={status} after {iterations} iterations, objective={FormatDouble(finalObjective)}");

            return new Solution(x.Clone(), finalObjective, status, iterations, watch.Elapsed);
        }

        private static bool PerformBacktrackingStep(
            QuadraticProblem problem,
            Vector<double> current,
            Vector<double> gradient,
            double penaltyWeight,
            double initialStep,
            double minStep,
            Vector<double> scratch,
            TextWriter writer,
            double tolerance,
            out double acceptedStepNorm,
            ref double currentPenaltyObjective)
        {
            acceptedStepNorm = 0.0;
            var step = initialStep;

            while (step >= minStep)
            {
                scratch.SetSubVector(0, scratch.Count, current);
                scratch -= gradient * step;
                Project(problem, scratch);

                var candidatePenalty = EvaluatePenalty(problem, scratch, penaltyWeight);

                if (candidatePenalty <= currentPenaltyObjective - tolerance * step * gradient.L2Norm())
                {
                    current.SetSubVector(0, scratch.Count, scratch);
                    acceptedStepNorm = (gradient * step).L2Norm();
                    currentPenaltyObjective = candidatePenalty;
                    return true;
                }

                step *= 0.5;
                writer?.WriteLine($"[quadprog]          backtracking: step reduced to {FormatDouble(step)} (penalty {FormatDouble(candidatePenalty)})");
            }

            return false;
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

        private static void EvaluateResiduals(
            QuadraticProblem problem,
            Vector<double> point,
            out double equalityNorm,
            out double inequalityViolation)
        {
            equalityNorm = 0.0;
            inequalityViolation = 0.0;

            if (problem.EqualityMatrix != null && problem.EqualityVector != null)
            {
                var equalityResidual = problem.EqualityMatrix * point - problem.EqualityVector;
                equalityNorm = equalityResidual.L2Norm();
            }

            if (problem.InequalityMatrix != null && problem.InequalityVector != null)
            {
                var inequalityResidual = problem.InequalityMatrix * point - problem.InequalityVector;
                inequalityViolation = inequalityResidual.Enumerate().Where(v => v > 0).DefaultIfEmpty(0.0).Max();
            }
        }

        private static double EvaluateObjective(QuadraticProblem problem, Vector<double> x)
        {
            var quadratic = 0.5 * x.DotProduct(problem.Q * x);
            var linear = problem.C.DotProduct(x);
            return quadratic + linear;
        }

        private static double EvaluatePenalty(QuadraticProblem problem, Vector<double> point, double penaltyWeight)
        {
            var value = EvaluateObjective(problem, point);

            if (problem.EqualityMatrix != null && problem.EqualityVector != null)
            {
                var residual = problem.EqualityMatrix * point - problem.EqualityVector;
                value += penaltyWeight * residual.DotProduct(residual);
            }

            if (problem.InequalityMatrix != null && problem.InequalityVector != null)
            {
                var residual = problem.InequalityMatrix * point - problem.InequalityVector;
                foreach (var r in residual)
                {
                    if (r > 0)
                    {
                        value += penaltyWeight * r * r;
                    }
                }
            }

            return problem.IsMinimisation ? value : -value;
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

        private static TextWriter ResolveWriter(SolverOptions options)
        {
            if (options == null)
            {
                return null;
            }

            if (options.DiagnosticsWriter != null)
            {
                return options.DiagnosticsWriter;
            }

            return options.Verbose ? Console.Out : null;
        }

        private static bool HasTimeLimitExpired(SolverOptions options, TimeSpan elapsed)
        {
            return options?.TimeLimit != null && elapsed >= options.TimeLimit.Value;
        }

        private static string FormatVector(Vector<double> vector)
        {
            if (vector == null)
            {
                return "[]";
            }

            return "[" + string.Join(", ", vector.Enumerate().Select(v => v.ToString("G6", CultureInfo.InvariantCulture))) + "]";
        }

        private static string FormatDouble(double value)
        {
            return value.ToString("G12", CultureInfo.InvariantCulture);
        }

        private static double EstimateInitialStep(QuadraticProblem problem, double penaltyWeight)
        {
            var normQ = problem.Q.L2Norm();
            var normEq = problem.EqualityMatrix?.L2Norm() ?? 0.0;
            var normIneq = problem.InequalityMatrix?.L2Norm() ?? 0.0;
            var lipschitz = normQ + 2 * penaltyWeight * (normEq * normEq + normIneq * normIneq);
            if (lipschitz <= 0)
            {
                lipschitz = 1.0;
            }

            return 1.0 / lipschitz;
        }
    }
}
