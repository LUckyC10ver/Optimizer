using System.IO;
using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Options used to configure the Sequential Quadratic Programming solver. The
    /// shape mirrors the configuration switches of the historic C++ toolbox so that
    /// existing code can be ported with minimal changes.
    /// </summary>
    public class SqpOptions : SolverOptions
    {
        /// <summary>
        /// Controls the verbosity of diagnostic logging. The semantics follow the
        /// original implementation where <c>0</c> disables logging and higher values
        /// gradually increase the amount of emitted information.
        /// </summary>
        public int Display { get; set; }

        /// <summary>
        /// Termination tolerance for the decision variables ("TolX"). When the
        /// maximum absolute step falls below this threshold the solver considers the
        /// iterate stationary.
        /// </summary>
        public double ArgumentTolerance { get; set; } = 1e-4;

        /// <summary>
        /// Termination tolerance on the projected gradient ("TolFun").
        /// </summary>
        public double ObjectiveTolerance { get; set; } = 1e-4;

        /// <summary>
        /// Tolerance used to check feasibility of the constraints.
        /// </summary>
        public double ConstraintToleranceOverride { get; set; } = 1e-6;

        /// <summary>
        /// Controls the choice of line-search strategy. Only a safeguarded
        /// backtracking search is implemented in the managed port, but the property is
        /// retained for API compatibility.
        /// </summary>
        public int LineSearch { get; set; }

        /// <summary>
        /// When true the solver validates user supplied gradients against finite
        /// difference approximations on the first iteration.
        /// </summary>
        public bool CheckGradients { get; set; }

        /// <summary>
        /// Maximum number of objective/constraint evaluations before the optimisation
        /// terminates. A value of zero lets the solver pick a sensible default.
        /// </summary>
        public int MaxFunctionEvaluations { get; set; }

        /// <summary>
        /// Minimum perturbation used when approximating derivatives numerically.
        /// </summary>
        public double FiniteDifferenceMinChange { get; set; } = 1e-9;

        /// <summary>
        /// Maximum perturbation used when approximating derivatives numerically.
        /// </summary>
        public double FiniteDifferenceMaxChange { get; set; } = 1e-6;

        /// <summary>
        /// Value that represents "infinity" in the original implementation. The
        /// property is not used directly but retained for completeness.
        /// </summary>
        public double LargeNumber { get; set; } = 1.7e38;

        /// <summary>
        /// Machine precision used when falling back to the legacy finite difference
        /// heuristics.
        /// </summary>
        public double MachineEpsilon { get; set; } = 1e-11;

        /// <summary>
        /// Optional writer that receives SQP level debug output.
        /// </summary>
        public TextWriter SqpLog { get; set; }

        /// <summary>
        /// Optional writer that receives QP level debug output.
        /// </summary>
        public TextWriter QpLog { get; set; }

        /// <summary>
        /// Maximum number of iterations granted to the QP sub-problem. A value of zero
        /// delegates the choice to the quadratic solver.
        /// </summary>
        public int MaxQuadraticIterations { get; set; } = 1000;

        /// <summary>
        /// Damping factor applied during merit-function based line search.
        /// </summary>
        public double ArmijoFactor { get; set; } = 1e-4;

        /// <summary>
        /// Reduction factor used when shrinking the trial step during backtracking.
        /// </summary>
        public double LineSearchShrink { get; set; } = 0.5;

        /// <summary>
        /// Maximum number of backtracking steps per iteration.
        /// </summary>
        public int MaxLineSearchSteps { get; set; } = 20;

        /// <summary>
        /// Initial penalty multiplier for the merit function.
        /// </summary>
        public double InitialPenalty { get; set; } = 10.0;

        /// <summary>
        /// Lower bound on the step length. When the backtracking search falls below
        /// this threshold the solver aborts and returns the best point obtained so far.
        /// </summary>
        public double MinimumStepLength { get; set; } = 1e-6;
    }
}
