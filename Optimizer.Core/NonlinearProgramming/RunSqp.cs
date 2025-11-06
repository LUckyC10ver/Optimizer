using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Public entry point that mirrors the classic C++ signature "runSqp".
    /// </summary>
    public static class RunSqp
    {
        /// <summary>
        /// Executes the simplified SQP-style optimisation routine.
        /// </summary>
        /// <param name="xout">Vector that will receive the final solution.</param>
        /// <param name="info">Structure that will be populated with diagnostic information.</param>
        /// <param name="lambda">Lagrange multiplier estimates. The vector is resized as needed.</param>
        /// <param name="act_ind">Indices of active constraints at the final iterate.</param>
        /// <param name="obj">Objective function delegate. Must return the function value for the supplied point.</param>
        /// <param name="lb">Lower bounds for the decision variables. May be <c>null</c>.</param>
        /// <param name="ub">Upper bounds for the decision variables. May be <c>null</c>.</param>
        /// <param name="Aequ">Matrix of linear equality constraints.</param>
        /// <param name="bequ">Right-hand side of the linear equality constraints.</param>
        /// <param name="Aieq">Matrix of linear inequality constraints.</param>
        /// <param name="bieq">Right-hand side of the linear inequality constraints.</param>
        /// <param name="X_orig">Initial point for the optimisation.</param>
        /// <param name="con">Optional nonlinear constraints delegate. May be <c>null</c>.</param>
        /// <param name="opt">Solver options. If <c>null</c> a default configuration is used.</param>
        /// <param name="Hess">Optional Hessian approximation parameter. Currently ignored but kept for signature compatibility.</param>
        public static void runSqp(
            ref Vector<double> xout,
            ref SqpInfo info,
            ref Vector<double> lambda,
            ref List<int> act_ind,
            Func<Vector<double>, double> obj,
            Vector<double> lb,
            Vector<double> ub,
            Matrix<double> Aequ,
            Vector<double> bequ,
            Matrix<double> Aieq,
            Vector<double> bieq,
            Vector<double> X_orig,
            Func<Vector<double>, ConstraintEvaluation> con,
            SqpOptions opt = null,
            Matrix<double> Hess = null)
        {
            if (obj == null)
            {
                throw new OptimizationException("Objective function delegate cannot be null.");
            }

            if (X_orig == null)
            {
                throw new OptimizationException("X_orig (initial guess) cannot be null.");
            }

            var dimension = X_orig.Count;
            xout ??= Vector<double>.Build.Dense(dimension);

            if (xout.Count != dimension)
            {
                throw new OptimizationException("xout must match the dimension of the initial guess.");
            }

            var effectiveOptions = opt ?? new SqpOptions();
            var solver = new SqpSolver(obj, con, Aequ, bequ, Aieq, bieq, lb, ub, effectiveOptions);
            solver.Solve(X_orig, xout);

            // Evaluate one last time to populate diagnostics and auxiliary data structures.
            solver.EvaluateCompositeObjective(xout, out var rawObjective, out var violation);

            info = info ?? new SqpInfo();
            info.ObjectiveValue = rawObjective;
            info.ConstraintViolation = violation;
            info.IterationCount = solver.IterationCount;
            info.FunctionEvaluations = solver.EvaluationCount;
            info.GradientNorm = solver.LastGradientNorm;
            info.Status = violation <= effectiveOptions.Tolerance
                ? "Converged"
                : "Finished without satisfying tolerance";

            // Populate lambda and active set outputs for compatibility. We treat all multipliers as zero
            // and simply report the indices of nearly active linear inequalities.
            var equalityCount = (Aequ?.RowCount ?? 0);
            var inequalityCount = (Aieq?.RowCount ?? 0);
            var nonlinearCount = con?.Invoke(xout)?.Values?.Count ?? 0;
            var totalConstraints = equalityCount + inequalityCount + nonlinearCount;

            lambda = Vector<double>.Build.Dense(totalConstraints);
            act_ind = act_ind ?? new List<int>();
            act_ind.Clear();

            for (var i = 0; i < equalityCount; i++)
            {
                act_ind.Add(i);
            }

            var tolerance = effectiveOptions.Tolerance;

            if (Aieq != null && bieq != null)
            {
                var residual = Aieq * xout - bieq;
                for (var i = 0; i < residual.Count; i++)
                {
                    if (Math.Abs(residual[i]) <= tolerance)
                    {
                        act_ind.Add(equalityCount + i);
                    }
                }
            }

            if (con != null)
            {
                var evaluation = con(xout) ?? ConstraintEvaluation.Empty;
                if (evaluation.Values != null)
                {
                    for (var i = 0; i < evaluation.Values.Count; i++)
                    {
                        var value = evaluation.Values[i];
                        var index = equalityCount + inequalityCount + i;

                        if (i < evaluation.EqualityCount)
                        {
                            if (!act_ind.Contains(index))
                            {
                                act_ind.Add(index);
                            }
                        }
                        else if (Math.Abs(value) <= tolerance)
                        {
                            act_ind.Add(index);
                        }
                    }
                }
            }

            // Hess parameter intentionally unused - retained for compatibility with the original signature.
            _ = Hess;
        }
    }
}
