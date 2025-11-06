using System.Text;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Diagnostic information returned by the SQP solver. The structure mirrors the
    /// legacy toolbox so that callers can reuse existing post processing code.
    /// </summary>
    public class SqpInfo
    {
        public double ObjectiveValue { get; set; }

        public int SqpCount { get; set; }

        public int FunctionCount { get; set; }

        public int GradientCount { get; set; }

        public double StepLength { get; set; } = 1.0;

        public string QpStatus { get; set; } = string.Empty;

        public string Status { get; set; } = string.Empty;

        public override string ToString()
        {
            var builder = new StringBuilder();
            builder.AppendLine($"SQP iterations: {SqpCount}");
            builder.AppendLine($"Function evaluations: {FunctionCount}");
            builder.AppendLine($"Gradient evaluations: {GradientCount}");
            builder.AppendLine($"Last step length: {StepLength}");
            if (!string.IsNullOrWhiteSpace(QpStatus))
            {
                builder.AppendLine($"QP status: {QpStatus}");
            }

            if (!string.IsNullOrWhiteSpace(Status))
            {
                builder.AppendLine($"SQP status: {Status}");
            }

            builder.AppendLine($"Objective value: {ObjectiveValue}");
            return builder.ToString();
        }
    }
}
