using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Facade mirroring the historical SQP entry point while exposing convenient helpers for managed callers.
    /// </summary>
    public static class RunSqp
    {
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
                throw new OptimizationException("X_orig cannot be null.");
            }

            xout ??= X_orig.Clone();
            info ??= new SqpInfo();
            act_ind ??= new List<int>();

            var problem = new NonlinearProblem(
                obj,
                null,
                Array.Empty<NonlinearConstraint>(),
                X_orig,
                lb,
                ub,
                Aequ,
                bequ,
                Aieq,
                bieq);

            var solver = new SequentialQuadraticProgrammingSolver();
            var solution = solver.Solve(problem, con, opt, info, out var multipliers, out var activeSet);

            xout = solution.Clone();
            lambda = multipliers;
            act_ind.Clear();
            act_ind.AddRange(activeSet);

            // Hess parameter is retained for signature compatibility.
            _ = Hess;
        }

        public static Solution Solve(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IEnumerable<NonlinearConstraint> constraints,
            Vector<double> initialGuess,
            SqpOptions options,
            out SqpInfo info)
        {
            if (objective == null)
            {
                throw new OptimizationException("Objective delegate must not be null.");
            }

            if (initialGuess == null)
            {
                throw new OptimizationException("An initial guess is required.");
            }

            var constraintList = constraints != null
                ? new List<NonlinearConstraint>(constraints)
                : new List<NonlinearConstraint>();

            var problem = new NonlinearProblem(
                objective,
                gradient,
                constraintList,
                initialGuess);

            var solver = new SequentialQuadraticProgrammingSolver();
            info = new SqpInfo();
            var solutionVector = solver.Solve(problem, null, options, info, out _, out _);
            return new Solution(solutionVector, info.ObjectiveValue, SolverResultStatus.Optimal, info.SqpCount, TimeSpan.Zero);
        }
    }
}
