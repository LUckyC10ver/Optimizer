using System.Collections.Generic;
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

            return new Solution
            {
                Status = SolverResultStatus.NotImplemented,
                Message = "Branch-and-bound solver is not yet implemented.",
                OptimalPoint = problem.BaseProblem?.C?.Clone()
            };
        }

        public Solution Solve(LinearProblem problem, IEnumerable<int> integerIndices, SolverOptions options = null)
        {
            var mixedProblem = new MixedIntegerProblem(problem, integerIndices);
            return Solve(mixedProblem, options);
        }
    }
}
