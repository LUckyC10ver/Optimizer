using System;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Implements a lightweight penalty-based search that mimics the interface of the legacy SQP entry point.
    /// </summary>
    internal sealed class SqpSolver
    {
        private readonly SqpOptions _options;
        private readonly Func<Vector<double>, double> _objective;
        private readonly Func<Vector<double>, ConstraintEvaluation> _constraints;
        private readonly Matrix<double> _linearEqualities;
        private readonly Vector<double> _linearEqualityRhs;
        private readonly Matrix<double> _linearInequalities;
        private readonly Vector<double> _linearInequalityRhs;
        private readonly Vector<double> _lowerBounds;
        private readonly Vector<double> _upperBounds;
        private int _evaluationCounter;

        public int IterationCount { get; private set; }

        public int EvaluationCount => _evaluationCounter;

        public double LastGradientNorm { get; private set; }

        public SqpSolver(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, ConstraintEvaluation> constraints,
            Matrix<double> linearEqualities,
            Vector<double> linearEqualityRhs,
            Matrix<double> linearInequalities,
            Vector<double> linearInequalityRhs,
            Vector<double> lowerBounds,
            Vector<double> upperBounds,
            SqpOptions options)
        {
            _objective = objective ?? throw new OptimizationException("Objective function must be provided.");
            _constraints = constraints;
            _linearEqualities = linearEqualities;
            _linearEqualityRhs = linearEqualityRhs;
            _linearInequalities = linearInequalities;
            _linearInequalityRhs = linearInequalityRhs;
            _lowerBounds = lowerBounds;
            _upperBounds = upperBounds;
            _options = options ?? new SqpOptions();
        }

        public void Solve(Vector<double> initialGuess, Vector<double> result)
        {
            if (initialGuess == null)
            {
                throw new OptimizationException("An initial guess must be supplied (X_orig cannot be null).");
            }

            if (result == null)
            {
                throw new ArgumentNullException(nameof(result));
            }

            if (initialGuess.Count != result.Count)
            {
                throw new OptimizationException("Initial guess and result vectors must share the same dimension.");
            }

            var current = initialGuess.Clone();
            var best = current.Clone();
            _evaluationCounter = 0;
            var bestValue = EvaluateCompositeObjective(current, out var bestObjective, out var bestViolation);

            var gradient = Vector<double>.Build.Dense(current.Count);
            var scratch = Vector<double>.Build.Dense(current.Count);
            IterationCount = 0;

            for (var iteration = 0; iteration < _options.MaxIterations; iteration++)
            {
                var gradientNorm = ApproximateGradient(current, gradient);
                LastGradientNorm = gradientNorm;

                if (gradientNorm < _options.Tolerance && bestViolation < _options.Tolerance)
                {
                    IterationCount = iteration + 1;
                    break;
                }

                scratch.SetSubVector(0, current.Count, current);
                scratch = scratch - gradient.Multiply(_options.StepSize);

                ProjectToBounds(scratch);

                var candidateValue = EvaluateCompositeObjective(scratch, out var candidateObjective, out var candidateViolation);

                if (candidateValue < bestValue)
                {
                    bestValue = candidateValue;
                    bestObjective = candidateObjective;
                    bestViolation = candidateViolation;
                    best.SetSubVector(0, scratch.Count, scratch);
                }

                current.SetSubVector(0, scratch.Count, scratch);

                _options.ProgressCallback?.Invoke(current, candidateValue, candidateObjective, candidateViolation);

                IterationCount = iteration + 1;
            }

            result.SetSubVector(0, best.Count, best);
        }

        public double EvaluateCompositeObjective(Vector<double> x, out double rawObjective, out double violation)
        {
            _evaluationCounter++;
            rawObjective = _objective(x);
            violation = 0.0;
            var penalty = 0.0;

            if (_linearEqualities != null && _linearEqualityRhs != null)
            {
                var residual = _linearEqualities * x - _linearEqualityRhs;
                for (var i = 0; i < residual.Count; i++)
                {
                    var value = residual[i];
                    penalty += value * value;
                    violation = Math.Max(violation, Math.Abs(value));
                }
            }

            if (_linearInequalities != null && _linearInequalityRhs != null)
            {
                var residual = _linearInequalities * x - _linearInequalityRhs;
                for (var i = 0; i < residual.Count; i++)
                {
                    var value = Math.Max(0.0, residual[i]);
                    penalty += value * value;
                    violation = Math.Max(violation, value);
                }
            }

            if (_constraints != null)
            {
                var evaluation = _constraints(x) ?? ConstraintEvaluation.Empty;
                if (evaluation.Values != null)
                {
                    for (var i = 0; i < evaluation.Values.Count; i++)
                    {
                        var value = evaluation.Values[i];
                        if (i < evaluation.EqualityCount)
                        {
                            penalty += value * value;
                            violation = Math.Max(violation, Math.Abs(value));
                        }
                        else
                        {
                            var clipped = Math.Max(0.0, value);
                            penalty += clipped * clipped;
                            violation = Math.Max(violation, clipped);
                        }
                    }
                }
            }

            return rawObjective + _options.PenaltyWeight * penalty;
        }

        public double ApproximateGradient(Vector<double> point, Vector<double> gradient)
        {
            var step = _options.FiniteDifferenceStep;
            var normSquared = 0.0;

            for (var i = 0; i < point.Count; i++)
            {
                var saved = point[i];
                point[i] = saved + step;
                var forward = EvaluateCompositeObjective(point, out _, out _);
                point[i] = saved - step;
                var backward = EvaluateCompositeObjective(point, out _, out _);
                point[i] = saved;

                var derivative = (forward - backward) / (2.0 * step);
                gradient[i] = derivative;
                normSquared += derivative * derivative;
            }

            return Math.Sqrt(normSquared);
        }

        private void ProjectToBounds(Vector<double> vector)
        {
            if (_lowerBounds != null)
            {
                for (var i = 0; i < vector.Count && i < _lowerBounds.Count; i++)
                {
                    vector[i] = Math.Max(vector[i], _lowerBounds[i]);
                }
            }

            if (_upperBounds != null)
            {
                for (var i = 0; i < vector.Count && i < _upperBounds.Count; i++)
                {
                    vector[i] = Math.Min(vector[i], _upperBounds[i]);
                }
            }
        }
    }
}
