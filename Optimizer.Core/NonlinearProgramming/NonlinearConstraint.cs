using System;
using MathNet.Numerics.LinearAlgebra;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Describes a nonlinear constraint via delegates.
    /// </summary>
    public sealed class NonlinearConstraint
    {
        public NonlinearConstraint(
            Func<Vector<double>, double> function,
            ConstraintKind kind,
            Func<Vector<double>, Vector<double>> gradient = null)
        {
            Function = function ?? throw new ArgumentNullException(nameof(function));
            Kind = kind;
            Gradient = gradient;
        }

        public Func<Vector<double>, double> Function { get; }

        public Func<Vector<double>, Vector<double>> Gradient { get; }

        public ConstraintKind Kind { get; }
    }

    public enum ConstraintKind
    {
        Equality,
        LessOrEqual,
        GreaterOrEqual
    }
}
