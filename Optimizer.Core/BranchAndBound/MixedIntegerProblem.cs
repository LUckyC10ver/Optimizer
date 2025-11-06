using System.Collections.Generic;
using Optimizer.Core.LinearProgramming;

namespace Optimizer.Core.BranchAndBound
{
    public class MixedIntegerProblem
    {
        public LinearProblem BaseProblem { get; }

        public ISet<int> IntegerIndices { get; }

        public MixedIntegerProblem(LinearProblem baseProblem, IEnumerable<int> integerIndices)
        {
            BaseProblem = baseProblem;
            IntegerIndices = integerIndices != null ? new HashSet<int>(integerIndices) : new HashSet<int>();
        }
    }
}
