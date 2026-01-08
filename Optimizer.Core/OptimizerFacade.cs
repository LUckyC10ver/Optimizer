using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.BranchAndBound;
using Optimizer.Core.Common;
using Optimizer.Core.LinearProgramming;
using Optimizer.Core.NonlinearProgramming;
using Optimizer.Core.QuadraticProgramming;
using RunSqpFacade = Optimizer.Core.NonlinearProgramming.RunSqp;

namespace Optimizer.Core
{
    public static class Optimizer
    {
        public static int quadprog(
            ref double[] x,
            ref double[] lambda,
            ref int[] act_ind,
            ref double[][] T,
            double[][] H,
            double[] f,
            double[][] Aieq,
            double[] bieq,
            double[][] Aeq,
            double[] beq,
            double[] lbd,
            double[] ubd,
            double[] x0,
            int verbosity = 0,
            double EPS = 1.0e-16,
            double BIG = 1.0e+99,
            int MAXITER = 1000,
            byte[] cdbg = default)
        {
            var solver = new QuadraticProgrammingSolver();
            var matrixH = Matrix<double>.Build.DenseOfRowArrays(H ?? Array.Empty<double[]>());
            var vectorF = Vector<double>.Build.DenseOfArray(f ?? Array.Empty<double>());
            var matrixIeq = Aieq != null ? Matrix<double>.Build.DenseOfRowArrays(Aieq) : null;
            var vectorIeq = bieq != null ? Vector<double>.Build.DenseOfArray(bieq) : null;
            var matrixEq = Aeq != null ? Matrix<double>.Build.DenseOfRowArrays(Aeq) : null;
            var vectorEq = beq != null ? Vector<double>.Build.DenseOfArray(beq) : null;
            var lower = lbd != null ? Vector<double>.Build.DenseOfArray(lbd) : null;
            var upper = ubd != null ? Vector<double>.Build.DenseOfArray(ubd) : null;
            var guess = x0 != null ? Vector<double>.Build.DenseOfArray(x0) : null;

            var xVector = x != null ? Vector<double>.Build.DenseOfArray(x) : Vector<double>.Build.Dense(vectorF.Count);
            var lambdaVector = lambda != null ? Vector<double>.Build.DenseOfArray(lambda) : Vector<double>.Build.Dense(0);
            var actList = act_ind != null ? new List<int>(act_ind) : new List<int>();
            var tableau = Matrix<double>.Build.Dense(0, 0);

            var solution = solver.Solve(
                ref xVector,
                ref lambdaVector,
                ref actList,
                ref tableau,
                matrixH,
                vectorF,
                matrixIeq,
                vectorIeq,
                matrixEq,
                vectorEq,
                lower,
                upper,
                guess,
                verbosity,
                EPS,
                BIG,
                MAXITER);

            x = xVector?.ToArray();
            lambda = lambdaVector?.ToArray();
            act_ind = actList.ToArray();
            T = tableau.RowCount > 0 ? tableau.ToRowArrays() : Array.Empty<double[]>();

            return solution?.Iterations ?? 0;
        }

        public static Solution LinProg(double[,] a, double[] b, double[] c, SolverOptions options = null)
        {
            var solver = new LinearProgrammingSolver();
            return solver.Solve(a, b, c, options);
        }

        public static Solution QuadProg(double[,] q, double[] c, double[,] a, double[] b, SolverOptions options = null)
        {
            var solver = new QuadraticProgrammingSolver();
            var matrixQ = Matrix<double>.Build.DenseOfArray(q ?? throw new OptimizationException("Quadratic matrix Q cannot be null."));
            var vectorC = Vector<double>.Build.DenseOfArray(c ?? throw new OptimizationException("Linear vector c cannot be null."));
            var matrixA = a != null ? Matrix<double>.Build.DenseOfArray(a) : null;
            var vectorB = b != null ? Vector<double>.Build.DenseOfArray(b) : null;
            var problem = new QuadraticProblem(matrixQ, vectorC, matrixA, vectorB);
            return solver.Solve(problem, options);
        }

        public static Solution QuadProg(
            double[,] q,
            double[] c,
            double[,] inequalityMatrix,
            double[] inequalityVector,
            double[,] equalityMatrix,
            double[] equalityVector,
            double[] lowerBounds,
            double[] upperBounds,
            double[] initialGuess,
            SolverOptions options = null)
        {
            var solver = new QuadraticProgrammingSolver();
            var matrixQ = Matrix<double>.Build.DenseOfArray(q ?? throw new OptimizationException("Quadratic matrix Q cannot be null."));
            var vectorC = Vector<double>.Build.DenseOfArray(c ?? throw new OptimizationException("Linear vector c cannot be null."));
            var matrixIneq = inequalityMatrix != null ? Matrix<double>.Build.DenseOfArray(inequalityMatrix) : null;
            var vectorIneq = inequalityVector != null ? Vector<double>.Build.DenseOfArray(inequalityVector) : null;
            var matrixEq = equalityMatrix != null ? Matrix<double>.Build.DenseOfArray(equalityMatrix) : null;
            var vectorEq = equalityVector != null ? Vector<double>.Build.DenseOfArray(equalityVector) : null;
            var lower = lowerBounds != null ? Vector<double>.Build.DenseOfArray(lowerBounds) : null;
            var upper = upperBounds != null ? Vector<double>.Build.DenseOfArray(upperBounds) : null;
            var guess = initialGuess != null ? Vector<double>.Build.DenseOfArray(initialGuess) : null;

            var problem = new QuadraticProblem(
                matrixQ,
                vectorC,
                matrixIneq,
                vectorIneq,
                true,
                matrixEq,
                vectorEq,
                lower,
                upper,
                guess);

            return solver.Solve(problem, options);
        }

        public static Solution SQP(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IEnumerable<NonlinearConstraint> constraints,
            double[] initialGuess,
            SqpOptions options = null)
        {
            var initialVector = initialGuess != null ? Vector<double>.Build.DenseOfArray(initialGuess) : null;
            return RunSqp(objective, gradient, constraints, initialVector, options);
        }

        public static Solution RunSqp(
            Func<Vector<double>, double> objective,
            Func<Vector<double>, Vector<double>> gradient,
            IEnumerable<NonlinearConstraint> constraints,
            Vector<double> initialGuess,
            SqpOptions options = null)
        {
            return RunSqpFacade.Solve(objective, gradient, constraints, initialGuess, options, out _);
        }

        public static Solution BranchAndBound(double[,] a, double[] b, double[] c, IEnumerable<int> integerIndices, SolverOptions options = null)
        {
            var matrix = Matrix<double>.Build.DenseOfArray(a ?? throw new OptimizationException("Matrix A cannot be null."));
            var vectorB = Vector<double>.Build.DenseOfArray(b ?? throw new OptimizationException("Vector b cannot be null."));
            var vectorC = Vector<double>.Build.DenseOfArray(c ?? throw new OptimizationException("Vector c cannot be null."));
            var linearProblem = new LinearProblem(matrix, vectorB, vectorC);
            var solver = new BranchAndBoundSolver();
            return solver.Solve(linearProblem, integerIndices, options);
        }
    }
}
