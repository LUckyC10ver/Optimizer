using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;
using Optimizer.Core.QuadraticProgramming;

namespace Optimizer.Core.NonlinearProgramming
{
    public class SequentialQuadraticProgrammingSolver
    {
        private readonly QuadraticProgrammingSolver _quadraticSolver = new QuadraticProgrammingSolver();

        public Solution Solve(NonlinearProblem problem, SqpOptions options = null)
        {
            if (problem == null)
            {
                throw new OptimizationException("Nonlinear problem cannot be null.");
            }

            options ??= new SqpOptions();

            return new Solution
            {
                Status = SolverResultStatus.NotImplemented,
                Message = "SQP solver is not yet implemented.",
                OptimalPoint = problem.InitialGuess?.Clone()
            };
        }

        public Solution Solve(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IEnumerable<NonlinearConstraint> constraints,
            Vector<double> initialGuess,
            SqpOptions options = null)
        {
            var constraintList = constraints != null ? new List<NonlinearConstraint>(constraints) : new List<NonlinearConstraint>();
            var problem = new NonlinearProblem(objective, gradient, constraintList, initialGuess, initialGuess?.Count ?? 0);
            return Solve(problem, options);
        }
    }
}
