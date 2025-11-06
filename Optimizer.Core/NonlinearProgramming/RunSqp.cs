using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.NonlinearProgramming
{
    public static class RunSqp
    {
        public static Solution Solve(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IEnumerable<NonlinearConstraint> constraints,
            Vector<double> initialGuess,
            SqpOptions options = null)
        {
            var solver = new SequentialQuadraticProgrammingSolver();
            return solver.Solve(objective, gradient, constraints, initialGuess, options);
        }
    }
}
