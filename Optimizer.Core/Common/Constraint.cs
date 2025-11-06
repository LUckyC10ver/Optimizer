using System;
using MathNet.Numerics.LinearAlgebra;

namespace Optimizer.Core.Common
{
    public enum ConstraintType
    {
        LessEqual,
        Equal,
        GreaterEqual
    }

    public abstract class Constraint
    {
        public ConstraintType Type { get; }

        protected Constraint(ConstraintType type)
        {
            Type = type;
        }

        public abstract bool IsSatisfied(Vector<double> point, double tolerance = 1e-9);

        protected static void ValidatePoint(Vector<double> point)
        {
            if (point == null)
            {
                throw new OptimizationException("Constraint evaluation requires a non-null point vector.");
            }
        }

        protected static bool IsLess(double value, double limit, double tolerance) => value <= limit + tolerance;

        protected static bool IsGreater(double value, double limit, double tolerance) => value >= limit - tolerance;
    }
}
