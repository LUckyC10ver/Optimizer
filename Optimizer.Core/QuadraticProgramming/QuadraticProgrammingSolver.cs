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
            var penaltyWeight = 100.0;
            var stepSize = 1.0 / Math.Max(problem.Q.RowCount, 1);

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
            var watch = Stopwatch.StartNew();
            var iterations = 0;
            var status = SolverResultStatus.IterationLimit;

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

                var step = gradient * stepSize;
                var stepNorm = step.L2Norm();
                x -= step;
                Project(problem, x);

                writer?.WriteLine(
                    $"[quadprog]          step={FormatDouble(stepNorm)} -> new x={FormatVector(x)}");
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
    }
}
