namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Diagnostic information reported by the SQP routine.
    /// </summary>
    public sealed class SqpInfo
    {
        public double ObjectiveValue { get; set; }

        public int SqpCount { get; set; }

        public int FunCount { get; set; }

        public int GradCount { get; set; }

        public double StepLength { get; set; }

        public string How { get; set; } = string.Empty;

        public string HowQP { get; set; } = string.Empty;

        public double ConstraintViolation { get; set; }

        public int IterationCount { get; set; }

        public double GradientNorm { get; set; }

        public string Status { get; set; } = string.Empty;
    }
}
