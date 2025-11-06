using System;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    public class NonlinearConstraint : Constraint
    {
        public Func<Vector<double>, double> Function { get; }

        public Func<Vector<double>, Vector<double>> Gradient { get; }

        public NonlinearConstraint(Func<Vector<double>, double> function, ConstraintType type, Func<Vector<double>, Vector<double>> gradient = null)
            : base(type)
        {
            Function = function;
            Gradient = gradient;
        }

        public override bool IsSatisfied(Vector<double> point, double tolerance = 1e-9)
        {
            ValidatePoint(point);
            if (Function == null)
            {
                throw new OptimizationException("Nonlinear constraint function must be provided.");
            }

            var value = Function(point);
            return Type switch
            {
                ConstraintType.Equal => System.Math.Abs(value) <= tolerance,
                ConstraintType.LessEqual => value <= tolerance,
                ConstraintType.GreaterEqual => value >= -tolerance,
                _ => false
            };
        }
    }
}
