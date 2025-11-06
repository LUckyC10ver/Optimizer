using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.LinearProgramming
{
    public class LinearConstraint : Constraint
    {
        public Vector<double> Coefficients { get; }

        public double Value { get; }

        public LinearConstraint(Vector<double> coefficients, double value, ConstraintType type)
            : base(type)
        {
            Coefficients = coefficients;
            Value = value;
        }

        public override bool IsSatisfied(Vector<double> point, double tolerance = 1e-9)
        {
            ValidatePoint(point);
            var evaluation = Coefficients.DotProduct(point);
            return Type switch
            {
                ConstraintType.Equal => System.Math.Abs(evaluation - Value) <= tolerance,
                ConstraintType.LessEqual => evaluation <= Value + tolerance,
                ConstraintType.GreaterEqual => evaluation >= Value - tolerance,
                _ => false
            };
        }
    }
}
