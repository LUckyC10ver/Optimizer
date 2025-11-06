using System.IO;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Container for the configuration values exposed by the legacy SQP interface.
    /// </summary>
    public sealed class SqpOptions
    {
        public SqpOptions()
        {
        }

        public int Display { get; set; } = 0;

        public double TolArg { get; set; } = 1e-2;

        public double TolObj { get; set; } = 1e-4;

        public double TolCon { get; set; } = 1e-6;

        public int LineSearch { get; set; } = 0;

        public bool CheckGrad { get; set; } = false;

        public int MaxFunEvals { get; set; } = 0;

        public double DiffMinChange { get; set; } = 1e-9;

        public double DiffMaxChange { get; set; } = 1e-6;

        public double MaxDouble { get; set; } = 1.7e38;

        public double Eps { get; set; } = 1e-11;

        public TextWriter SqpLog { get; set; }
            = null;

        public TextWriter QpLog { get; set; }
            = null;

        public int MaxNumItersQP { get; set; } = 1_000;

        public double PenaltyWeight { get; set; } = 50.0;

        public double StepSize { get; set; } = 0.1;
    }
}
