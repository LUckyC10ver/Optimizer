using System;
using System.Collections.Generic;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;

namespace Optimizer.Core.LinearProgramming
{
    public class LinearProgrammingSolver
    {
        public Solution Solve(LinearProblem problem, SolverOptions options = null)
        {
            if (problem == null)
            {
                throw new OptimizationException("Linear programming problem cannot be null.");
            }

            options ??= new SolverOptions();
            var tolerance = options.Tolerance > 0 ? options.Tolerance : 1e-9;
            var constraintTolerance = options.ConstraintTolerance > 0 ? options.ConstraintTolerance : tolerance;

            var inequalityMatrix = problem.A ?? throw new OptimizationException("Constraint matrix A cannot be null.");
            var inequalityVector = problem.B ?? throw new OptimizationException("Constraint vector b cannot be null.");
            var objectiveVector = problem.C ?? throw new OptimizationException("Objective vector c cannot be null.");

            if (inequalityMatrix.RowCount != inequalityVector.Count)
            {
                throw new OptimizationException("The inequality constraint matrix row count must match the vector length.");
            }

            if (inequalityMatrix.ColumnCount != objectiveVector.Count)
            {
                throw new OptimizationException("The number of columns in A must equal the length of c.");
            }

            var equalityMatrix = problem.EqualityMatrix ?? Matrix<double>.Build.Dense(0, objectiveVector.Count);
            var equalityVector = problem.EqualityVector ?? Vector<double>.Build.Dense(equalityMatrix.RowCount);

            if (equalityMatrix.ColumnCount != objectiveVector.Count)
            {
                throw new OptimizationException("The equality constraint matrix must have the same number of columns as c.");
            }

            if (equalityMatrix.RowCount != equalityVector.Count)
            {
                throw new OptimizationException("The equality constraint matrix row count must match the equality vector length.");
            }

            var stopwatch = Stopwatch.StartNew();

            var objective = objectiveVector.Clone();
            if (!problem.IsMinimize)
            {
                objective = objective.Negate();
            }

            var result = SolveInternal(
                inequalityMatrix,
                inequalityVector,
                equalityMatrix,
                equalityVector,
                objective,
                tolerance);

            stopwatch.Stop();

            if (result.Status == SolverResultStatus.Optimal)
            {
                var objectiveValue = problem.C.DotProduct(result.OptimalPoint);
                result.OptimalValue = objectiveValue;

                var activeSet = BuildActiveSet(inequalityMatrix, inequalityVector, equalityMatrix, equalityVector, result.OptimalPoint, constraintTolerance);
                if (activeSet.Count > 0)
                {
                    result.Message = $"Active constraints: {string.Join(",", activeSet)}";
                }
            }

            result.SolveTime = stopwatch.Elapsed;
            return result;
        }

        public Solution Solve(double[,] a, double[] b, double[] c, SolverOptions options = null)
        {
            return Solve(a, b, c, null, null, options);
        }

        public Solution Solve(double[,] a, double[] b, double[] c, double[,] aeq, double[] beq, SolverOptions options = null)
        {
            var matrix = Matrix<double>.Build.DenseOfArray(a ?? throw new OptimizationException("Constraint matrix A cannot be null."));
            var vectorB = Vector<double>.Build.DenseOfArray(b ?? throw new OptimizationException("Constraint vector b cannot be null."));
            var vectorC = Vector<double>.Build.DenseOfArray(c ?? throw new OptimizationException("Objective vector c cannot be null."));
            Matrix<double> matrixAeq = aeq != null ? Matrix<double>.Build.DenseOfArray(aeq) : Matrix<double>.Build.Dense(0, vectorC.Count);
            Vector<double> vectorBeq = beq != null ? Vector<double>.Build.DenseOfArray(beq) : Vector<double>.Build.Dense(matrixAeq.RowCount);
            var problem = new LinearProblem(matrix, vectorB, vectorC, equalityMatrix: matrixAeq, equalityVector: vectorBeq);
            return Solve(problem, options);
        }

        public Solution SolveAsyncPlaceholder(LinearProblem problem, SolverOptions options = null)
        {
            throw new OptimizationException("Asynchronous linear programming solver is not implemented in this skeleton.");
        }

        private static Solution SolveInternal(
            Matrix<double> inequalityMatrix,
            Vector<double> inequalityVector,
            Matrix<double> equalityMatrix,
            Vector<double> equalityVector,
            Vector<double> objective,
            double tolerance)
        {
            var nvars = objective.Count;
            var nieq = inequalityVector.Count;
            var neq = equalityVector.Count;

            var rows = nieq + neq + 2;
            var cols = 2 * nvars + 1;
            var tableau = Matrix<double>.Build.Dense(rows, cols);

            for (int i = 0; i < nvars; i++)
            {
                tableau[0, i + 1] = -objective[i];
                tableau[0, i + nvars + 1] = objective[i];
            }

            var negativeInequalityIndices = new List<int>();
            var nextRow = 1;
            for (int i = 0; i < nieq; i++)
            {
                if (inequalityVector[i] >= 0.0)
                {
                    tableau[nextRow, 0] = inequalityVector[i];
                    for (int j = 0; j < nvars; j++)
                    {
                        var coefficient = inequalityMatrix[i, j];
                        tableau[nextRow, j + 1] = -coefficient;
                        tableau[nextRow, j + nvars + 1] = coefficient;
                    }

                    nextRow++;
                }
                else
                {
                    negativeInequalityIndices.Add(i);
                }
            }

            foreach (var index in negativeInequalityIndices)
            {
                tableau[nextRow, 0] = -inequalityVector[index];
                for (int j = 0; j < nvars; j++)
                {
                    var coefficient = inequalityMatrix[index, j];
                    tableau[nextRow, j + 1] = coefficient;
                    tableau[nextRow, j + nvars + 1] = -coefficient;
                }

                nextRow++;
            }

            for (int i = 0; i < neq; i++)
            {
                var sign = equalityVector[i] >= 0 ? 1.0 : -1.0;
                var rowIndex = nieq + i + 1;
                tableau[rowIndex, 0] = sign * equalityVector[i];
                for (int j = 0; j < nvars; j++)
                {
                    var coefficient = sign * equalityMatrix[i, j];
                    tableau[rowIndex, j + 1] = -coefficient;
                    tableau[rowIndex, j + nvars + 1] = coefficient;
                }
            }

            var m1 = nieq - negativeInequalityIndices.Count;
            var m2 = negativeInequalityIndices.Count;
            var m3 = neq;

            var izrov = new int[cols - 1];
            var iposv = new int[rows - 2];

            int icase;
            try
            {
                Simplex(tableau, (int)m1, (int)m2, (int)m3, out icase, izrov, iposv, tolerance);
            }
            catch (OptimizationException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new OptimizationException($"Simplex solver failed: {ex.Message}", ex);
            }

            if (icase == -1)
            {
                return new Solution { Status = SolverResultStatus.Infeasible };
            }

            if (icase == 1)
            {
                return new Solution { Status = SolverResultStatus.Unbounded };
            }

            var solutionArray = new double[nvars];
            for (int j = 1; j <= iposv.Length; j++)
            {
                var basisIndex = iposv[j - 1] - 1;
                if (basisIndex < 0)
                {
                    continue;
                }

                var value = tableau[j, 0];

                if (basisIndex < nvars)
                {
                    solutionArray[basisIndex] += value;
                }
                else if (basisIndex < 2 * nvars)
                {
                    solutionArray[basisIndex - nvars] -= value;
                }
            }

            var solutionVector = Vector<double>.Build.DenseOfArray(solutionArray);

            return new Solution
            {
                OptimalPoint = solutionVector,
                Status = SolverResultStatus.Optimal
            };
        }

        private static List<int> BuildActiveSet(
            Matrix<double> inequalityMatrix,
            Vector<double> inequalityVector,
            Matrix<double> equalityMatrix,
            Vector<double> equalityVector,
            Vector<double> point,
            double tolerance)
        {
            var active = new List<int>();

            var eqCount = equalityMatrix.RowCount;
            for (int i = 0; i < eqCount; i++)
            {
                active.Add(i);
            }

            if (inequalityMatrix.RowCount > 0)
            {
                var residual = inequalityMatrix * point - inequalityVector;
                for (int i = 0; i < residual.Count; i++)
                {
                    if (Math.Abs(residual[i]) <= tolerance)
                    {
                        active.Add(eqCount + i);
                    }
                }
            }

            return active;
        }

        private static void Simplex(
            Matrix<double> tableau,
            int m1,
            int m2,
            int m3,
            out int icase,
            int[] izrov,
            int[] iposv,
            double eps)
        {
            icase = 0;

            var m = tableau.RowCount - 2;
            var n = tableau.ColumnCount - 1;

            if (m != (m1 + m2 + m3))
            {
                throw new OptimizationException("Bad input constraint counts in simplex");
            }

            var l1 = new int[n];
            var l2 = new int[m];
            var l3 = new int[m];

            var nl1 = n;
            for (var k = 1; k <= n; k++)
            {
                l1[k - 1] = k;
                izrov[k - 1] = k;
            }

            var nl2 = m;
            for (var i = 1; i <= m; i++)
            {
                if (tableau[i, 0] < 0.0)
                {
                    throw new OptimizationException("Bad input tableau in simplex");
                }

                l2[i - 1] = i;
                iposv[i - 1] = n + i;
            }

            for (var i = 1; i <= m2; i++)
            {
                l3[i - 1] = 1;
            }

            var ir = 0;
            if (m2 + m3 > 0)
            {
                ir = 1;
                for (var k = 1; k <= n + 1; k++)
                {
                    var q1 = 0.0;
                    for (var i = m1 + 1; i <= m; i++)
                    {
                        q1 += tableau[i, k - 1];
                    }

                    tableau[m + 1, k - 1] = -q1;
                }

                while (true)
                {
                    Simp1(tableau, m + 1, l1, nl1, false, out var kp, out var bmax);

                    if (bmax <= eps && tableau[m + 1, 0] < -eps)
                    {
                        icase = -1;
                        return;
                    }

                    if (bmax <= eps && Math.Abs(tableau[m + 1, 0]) <= eps)
                    {
                        var m12 = m1 + m2 + 1;
                        if (m12 <= m)
                        {
                            for (var ip = m12; ip <= m; ip++)
                            {
                                if (iposv[ip - 1] == n + ip)
                                {
                                    Simp1(tableau, ip, l1, nl1, true, out kp, out bmax);
                                    if (bmax > eps)
                                    {
                                        goto PhaseOnePivot;
                                    }
                                }
                            }
                        }

                        ir = 0;
                        m12--;
                        if (m1 + 1 <= m12)
                        {
                            for (var i = m1 + 1; i <= m12; i++)
                            {
                                if (l3[i - m1 - 1] == 1)
                                {
                                    for (var k = 1; k <= n + 1; k++)
                                    {
                                        tableau[i, k - 1] = -tableau[i, k - 1];
                                    }
                                }
                            }
                        }

                        break;
                    }

                    Simp2(tableau, n, l2, nl2, out var ip, kp, out var q1, eps);
                    if (ip == 0)
                    {
                        icase = -1;
                        return;
                    }

PhaseOnePivot:
                    Simp3(tableau, m + 1, n, ip, kp);

                    if (iposv[ip - 1] >= (n + m1 + m2 + 1))
                    {
                        var isIndex = 1;
                        for (; isIndex <= nl1; isIndex++)
                        {
                            if (l1[isIndex - 1] == kp)
                            {
                                break;
                            }
                        }

                        nl1--;
                        for (var is = isIndex; is <= nl1; is++)
                        {
                            l1[is - 1] = l1[is];
                        }

                        tableau[m + 1, kp] += 1.0;

                        for (var i = 1; i <= m + 2; i++)
                        {
                            tableau[i - 1, kp] = -tableau[i - 1, kp];
                        }
                    }
                    else if (iposv[ip - 1] >= (n + m1 + 1))
                    {
                        var kh = iposv[ip - 1] - m1 - n;
                        if (l3[kh - 1] != 0)
                        {
                            l3[kh - 1] = 0;
                            tableau[m + 1, kp] += 1.0;
                            for (var i = 1; i <= m + 2; i++)
                            {
                                tableau[i - 1, kp] = -tableau[i - 1, kp];
                            }
                        }
                    }

                    var isValue = izrov[kp - 1];
                    izrov[kp - 1] = iposv[ip - 1];
                    iposv[ip - 1] = isValue;

                    if (ir == 0)
                    {
                        break;
                    }
                }
            }

            while (true)
            {
                Simp1(tableau, 0, l1, nl1, false, out var kp, out var bmax);

                if (bmax < eps)
                {
                    icase = 0;
                    return;
                }

                Simp2(tableau, n, l2, nl2, out var ip, kp, out var q1, eps);
                if (ip == 0)
                {
                    icase = 1;
                    return;
                }

                Simp3(tableau, m, n, ip, kp);

                var isValue = izrov[kp - 1];
                izrov[kp - 1] = iposv[ip - 1];
                iposv[ip - 1] = isValue;
            }
        }

        private static void Simp1(
            Matrix<double> tableau,
            int row,
            int[] list,
            int listLength,
            bool absolute,
            out int kp,
            out double bmax)
        {
            kp = list[0];
            bmax = tableau[row, kp];

            for (var k = 2; k <= listLength; k++)
            {
                var column = list[k - 1];
                var test = absolute
                    ? Math.Abs(tableau[row, column]) - Math.Abs(bmax)
                    : tableau[row, column] - bmax;

                if (test > 0.0)
                {
                    bmax = tableau[row, column];
                    kp = column;
                }
            }
        }

        private static void Simp2(
            Matrix<double> tableau,
            int n,
            int[] l2,
            int nl2,
            out int ip,
            int kp,
            out double q1,
            double eps)
        {
            ip = 0;
            q1 = 0.0;

            for (var iIndex = 1; iIndex <= nl2; iIndex++)
            {
                var row = l2[iIndex - 1];
                if (tableau[row, kp] < -eps)
                {
                    q1 = -tableau[row, 0] / tableau[row, kp];
                    ip = row;
                    for (var nextIndex = iIndex + 1; nextIndex <= nl2; nextIndex++)
                    {
                        var ii = l2[nextIndex - 1];
                        if (tableau[ii, kp] < -eps)
                        {
                            var q = -tableau[ii, 0] / tableau[ii, kp];
                            if (q < q1)
                            {
                                ip = ii;
                                q1 = q;
                            }
                            else if (Math.Abs(q - q1) <= eps)
                            {
                                for (var k = 1; k <= n; k++)
                                {
                                    var qp = -tableau[ip, k] / tableau[ip, kp];
                                    var q0 = -tableau[ii, k] / tableau[ii, kp];
                                    if (Math.Abs(q0 - qp) > eps)
                                    {
                                        if (q0 < qp)
                                        {
                                            ip = ii;
                                            q1 = q;
                                        }

                                        break;
                                    }
                                }
                            }
                        }
                    }

                    return;
                }
            }
        }

        private static void Simp3(
            Matrix<double> tableau,
            int i1,
            int k1,
            int ip,
            int kp)
        {
            var piv = 1.0 / tableau[ip, kp];

            for (var ii = 1; ii <= i1 + 1; ii++)
            {
                if (ii - 1 == ip)
                {
                    continue;
                }

                tableau[ii - 1, kp] *= piv;
                for (var kk = 1; kk <= k1 + 1; kk++)
                {
                    if (kk - 1 == kp)
                    {
                        continue;
                    }

                    tableau[ii - 1, kk - 1] -= tableau[ip, kk - 1] * tableau[ii - 1, kp];
                }
            }

            for (var kk = 1; kk <= k1 + 1; kk++)
            {
                if (kk - 1 == kp)
                {
                    continue;
                }

                tableau[ip, kk - 1] *= -piv;
            }

            tableau[ip, kp] = piv;
        }
    }
}
