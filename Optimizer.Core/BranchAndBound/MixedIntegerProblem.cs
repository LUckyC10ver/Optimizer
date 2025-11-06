using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;
using Optimizer.Core.LinearProgramming;

namespace Optimizer.Core.BranchAndBound
{
    public class MixedIntegerProblem
    {
        public LinearProblem BaseProblem { get; }

        public ISet<int> IntegerIndices { get; }

        public int VariableCount => BaseProblem?.C?.Count ?? 0;

        public MixedIntegerProblem(LinearProblem baseProblem, IEnumerable<int> integerIndices)
        {
            BaseProblem = baseProblem ?? throw new OptimizationException("Base linear problem cannot be null.");

            if (BaseProblem.C == null)
            {
                throw new OptimizationException("Objective vector c cannot be null for a mixed-integer problem.");
            }

            IntegerIndices = integerIndices != null
                ? new HashSet<int>(integerIndices.Where(i => i >= 0))
                : new HashSet<int>();

            foreach (var index in IntegerIndices)
            {
                if (index >= BaseProblem.C.Count)
                {
                    throw new OptimizationException($"Integer index {index} is outside the variable range.");
                }
            }
        }

        public Vector<double> GetInitialLowerBounds()
        {
            var dimension = VariableCount;
            if (dimension == 0)
            {
                return Vector<double>.Build.Dense(0);
            }

            var lower = Vector<double>.Build.Dense(dimension, double.NegativeInfinity);
            if (BaseProblem.LowerBounds != null)
            {
                if (BaseProblem.LowerBounds.Count != dimension)
                {
                    throw new OptimizationException("Lower bound vector length must match the number of variables.");
                }

                for (var i = 0; i < dimension; i++)
                {
                    lower[i] = BaseProblem.LowerBounds[i];
                }
            }

            return lower;
        }

        public Vector<double> GetInitialUpperBounds()
        {
            var dimension = VariableCount;
            if (dimension == 0)
            {
                return Vector<double>.Build.Dense(0);
            }

            var upper = Vector<double>.Build.Dense(dimension, double.PositiveInfinity);
            if (BaseProblem.UpperBounds != null)
            {
                if (BaseProblem.UpperBounds.Count != dimension)
                {
                    throw new OptimizationException("Upper bound vector length must match the number of variables.");
                }

                for (var i = 0; i < dimension; i++)
                {
                    upper[i] = BaseProblem.UpperBounds[i];
                }
            }

            return upper;
        }
    }
}
