using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;
using Optimizer.Core.LinearProgramming;

namespace Optimizer.Core.BranchAndBound
{
    /// <summary>
    /// Represents a mixed integer linear programming instance that reuses the linear relaxation infrastructure.
    /// </summary>
    public sealed class MixedIntegerProblem
    {
        public MixedIntegerProblem(LinearProblem relaxation, IEnumerable<int> integerIndices)
        {
            Relaxation = relaxation ?? throw new OptimizationException("A linear relaxation must be supplied.");
            IntegerIndices = new HashSet<int>(integerIndices ?? new int[0]);
        }

        public LinearProblem Relaxation { get; }

        public HashSet<int> IntegerIndices { get; }

        public MixedIntegerProblem WithAdditionalConstraint(Vector<double> coefficients, double value)
        {
            var currentA = Relaxation.A;
            var currentB = Relaxation.B;

            var newA = Matrix<double>.Build.Dense(currentA.RowCount + 1, currentA.ColumnCount);
            newA.SetSubMatrix(0, currentA.RowCount, 0, currentA.ColumnCount, currentA);
            newA.SetRow(currentA.RowCount, coefficients);

            var newB = Vector<double>.Build.Dense(currentB.Count + 1);
            newB.SetSubVector(0, currentB.Count, currentB);
            newB[currentB.Count] = value;

            var newProblem = new LinearProblem(newA, newB, Relaxation.C, Relaxation.IsMinimisation);
            return new MixedIntegerProblem(newProblem, IntegerIndices);
        }
    }
}
