using System;
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

        /// <summary>
        /// Unified tolerance accessor used by legacy call-sites.
        /// </summary>
        public double Tolerance
        {
            get => Math.Min(Math.Min(TolArg, TolObj), TolCon);
            set
            {
                TolArg = value;
                TolObj = value;
                TolCon = value;
            }
        }

        /// <summary>
        /// Convenience alias for the finite-difference step length.
        /// </summary>
        public double FiniteDifferenceStep
        {
            get => DiffMinChange;
            set
            {
                DiffMinChange = value;
                if (DiffMaxChange < value)
                {
                    DiffMaxChange = value;
                }
            }
        }

        /// <summary>
        /// Maximum number of SQP iterations allowed before termination.
        /// </summary>
        public int MaxIterations
        {
            get => MaxFunEvals;
            set => MaxFunEvals = value;
        }

        /// <summary>
        /// Optional hook that receives per-iteration diagnostics. The callback arguments are
        /// (iteration index, raw objective, maximum constraint violation, full SQP info snapshot).
        /// </summary>
        public Action<int, double, double, SqpInfo> ProgressCallback { get; set; }
            = null;

        /// <summary>
        /// Maintained for compatibility with earlier builds that expected a single-argument
        /// progress reporter. When supplied, the delegate is invoked alongside
        /// <see cref="ProgressCallback"/>.
        /// </summary>
        public Action<SqpInfo> ProgressReporter { get; set; }
            = null;
    }
}
