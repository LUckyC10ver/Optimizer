using System;
using MathNet.Numerics.LinearAlgebra;

namespace Optimizer.Core.Common
{
    /// <summary>
    /// Specifies the relation enforced by a constraint.
    /// </summary>
    public enum ConstraintType
    {
        LessOrEqual,
        Equal,
        GreaterOrEqual
    }

    /// <summary>
    /// Base abstraction for constraints used across the optimisation modules.
    /// </summary>
    public abstract class Constraint
    {
        protected Constraint(ConstraintType type)
        {
            Type = type;
        }

        public ConstraintType Type { get; }

        /// <summary>
        /// Evaluates whether a candidate point satisfies the constraint within the supplied tolerance.
        /// </summary>
        public abstract bool IsSatisfied(Vector<double> x, double tolerance);
    }

    /// <summary>
    /// Represents a single linear constraint of the form a·x (relation) value.
    /// </summary>
    public sealed class LinearConstraint : Constraint
    {
        public LinearConstraint(Vector<double> coefficients, double value, ConstraintType type)
            : base(type)
        {
            Coefficients = coefficients ?? throw new ArgumentNullException(nameof(coefficients));
            Value = value;
        }

        public Vector<double> Coefficients { get; }

        public double Value { get; }

        public override bool IsSatisfied(Vector<double> x, double tolerance)
        {
            var lhs = Coefficients.DotProduct(x);
            return Type switch
            {
                ConstraintType.LessOrEqual => lhs <= Value + tolerance,
                ConstraintType.Equal => Math.Abs(lhs - Value) <= tolerance,
                ConstraintType.GreaterOrEqual => lhs >= Value - tolerance,
                _ => false
            };
        }
    }
}
