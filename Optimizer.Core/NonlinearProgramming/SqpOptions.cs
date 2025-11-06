using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    public class SqpOptions : SolverOptions
    {
        public double GradientTolerance { get; set; } = 1e-6;

        public double StepSizeTolerance { get; set; } = 1e-6;

        public bool UseLineSearch { get; set; } = true;
    }
}
