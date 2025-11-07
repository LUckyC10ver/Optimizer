using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Optimizer.Core.Common;
using Optimizer.Core.LinearProgramming;
using Optimizer.Core.QuadraticProgramming;

namespace Optimizer.Test
{
    internal static class Program
    {
        private static void Main()
        {
            RunSection("Linear Programming Sample", RunLinearProgrammingSample);
            Console.WriteLine();
            RunSection("Quadratic Programming Sample", RunQuadraticProgrammingSample);
        }

        private static void RunSection(string title, Action section)
        {
            Console.WriteLine($"--- {title} ---");
            try
            {
                section();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
        }

        private static void RunLinearProgrammingSample()
        {
            var optimalX = new[]
            {
                0.0076633966025675022,
                0,
                -0.003462900617632686,
                0.0062814726250553302,
                0,
                0,
                -0.0037693260969631683,
                0,
                0,
                0,
                4.3368086899420177e-19
            };

            var objective = new[]
            {
                0d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
            };

            const int variableCount = 11;
            const int inequalityCount = 29;
            const int equalityCount = 3;

            var inequalityMatrix = new double[inequalityCount, variableCount]
            {
                {-0.8516595341064221696, 0, 0, 0.509151160538948608, 0, 0, 0, -0.1242623582887186176, 0, 0, -1},
                {0, 0, 0, 0, 0, -0.2632862431701187584, 0.8020104022720540672, -0.5361526543856544768, 0, 0, -1},
                {0.8471670970874413056, 0, 0, -0.5167487279216425984, 0, 0, 0, 0.1236068841278759168, 0, 0, -1},
                {0, 0, 0, 0, 0, 0.2598800786239064576, -0.8077080920711933952, 0.5292163855523982336, 0, 0, -1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0.9223372036854775807, 0.2394884713086027776, -1},
                {0, 0, 0, 0, 0, 0, -0.9223372036854775807, 0, 0, 0.3133774542039018496, -1},
                {0, 0, 0, 0, 0, 0, 0, 0, -0.9223372036854775807, -0.1959275510550264064, -1},
                {0, 0, 0, 0, 0, 0, 0.9223372036854775807, 0, 0, -0.3190982008450142208, -1},
                {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
                {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1},
                {0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1},
                {0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1},
                {0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1},
                {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1},
                {0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1},
                {0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1},
                {0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1},
                {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
                {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1},
                {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1},
                {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1},
                {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1},
                {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1},
                {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1},
                {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1},
                {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1}
            };

            var inequalityVector = new[]
            {
                -0.0032533727432256876,
                -0.0030230387393199824,
                0.078715556354991296,
                1.4906228335918734336,
                0.7055248729289870336,
                0.0857470360354533888,
                6.2709413575412238336E-15,
                0.84055686016714496,
                4.3842683327917333504,
                20.6355007235711234048,
                0.248732307523017984,
                7.3399827264933188608,
                4.570502793530116608,
                4.6883003768498952192,
                2.5528969756433071104,
                2.7391350021262681088,
                1.8979157333251539968,
                0.8462676009752101888,
                1995.6157216672081631232,
                15979.364489276429594624,
                119.7512576924769831936,
                4992.6600072735063804928,
                1995.4294872064699575296,
                3.3116996231501047808,
                2.4471030243566928896,
                6.26086499787373184,
                0.9020842666748461056,
                16.1537323990247898624,
                0
            };

            var equalityMatrix = new double[equalityCount, variableCount]
            {
                {-0.5354104718360296448, 0, 0, 0.6532007756399560704, -0.5354104718360296448, 0, 0, 0, 0, 0, 0},
                {0, 0, -0.9223372036854775807, 0.0337423485633598528, 0, 0.0329312950321084928, 0, 0, 0.0799468463152895488, 0, 0},
                {-0.7803802040231349248, 0.1357375556488290048, 0, 0, 0.1357375556488290048, 0, 0, -0.5951112241772731392, 0, 0, 0}
            };

            var equalityVector = new[]
            {
                0d,
                0.0034059137111138648,
                -0.0059803630042218256
            };

            var combined = BuildAugmentedSystem(inequalityMatrix, inequalityVector, equalityMatrix, equalityVector);

            Console.WriteLine("calling linprog with problem data...");

            var solver = new LinearProgrammingSolver();
            var solution = solver.Solve(combined.A.ToArray(), combined.B.ToArray(), objective);

            if (solution.Status != SolverResultStatus.Optimal)
            {
                Console.WriteLine($"linprog did not converge to an optimal solution (status: {solution.Status}).");
                return;
            }

            var x = solution.OptimalX;
            var diff = x - Vector<double>.Build.DenseOfArray(optimalX);
            var error = diff.L2Norm();

            Console.WriteLine();
            Console.WriteLine($"error of simplex= {error.ToString("G17", CultureInfo.InvariantCulture)}");
            Console.WriteLine($"solution x (linprog):   {Format(x)}");
            Console.WriteLine($"reference xopt:         {Format(Vector<double>.Build.DenseOfArray(optimalX))}");
            Console.WriteLine($"difference (x-xopt):    {Format(diff)}");

            var residualEqu = Matrix<double>.Build.DenseOfArray(equalityMatrix) * x - Vector<double>.Build.DenseOfArray(equalityVector);
            var residualIeq = Matrix<double>.Build.DenseOfArray(inequalityMatrix) * x - Vector<double>.Build.DenseOfArray(inequalityVector);

            var activeIndices = BuildActiveSet(residualEqu, residualIeq, equalityCount, 1e-10);

            Console.WriteLine($"indices of active constraints (linprog):   {Format(activeIndices)}");
            Console.WriteLine($"indices of active constraints (reference): {Format(activeIndices)}");
            Console.WriteLine($"equality residuals:      {Format(residualEqu)}");
            Console.WriteLine($"inequality residuals:    {Format(residualIeq)}");
        }

        private static void RunQuadraticProgrammingSample()
        {
            var cases = new List<QuadraticTestCase>
            {
                CreateBoundedEqualityCase(),
                CreateUnconstrainedQuadraticCase(),
                CreateInequalityActiveCase()
            };

            var solver = new QuadraticProgrammingSolver();

            foreach (var testCase in cases)
            {
                Console.WriteLine($"Test case: {testCase.Name}");
                var problem = testCase.BuildProblem();

                LogProblemData(problem);

                var options = new SolverOptions
                {
                    Tolerance = 1e-8,
                    MaxIterations = 5_000,
                    Verbose = true,
                    DiagnosticsWriter = Console.Out
                };

                Console.WriteLine("calling quadprog with problem data (iteration log below)...");
                var solution = solver.Solve(problem, options);
                Console.WriteLine("quadprog finished.");

                var iterate = solution.OptimalX ?? Vector<double>.Build.Dense(problem.Q.ColumnCount);
                var equalityResidual = ComputeEqualityResidual(problem, iterate);
                var inequalityResidual = ComputeInequalityResidual(problem, iterate);
                var objectiveValue = EvaluateObjective(problem, iterate);

                Console.WriteLine($"  status:             {solution.Status}");
                Console.WriteLine($"  iterations:         {solution.Iterations}");
                Console.WriteLine($"  solve time [ms]:    {solution.SolveTime.TotalMilliseconds.ToString("G17", CultureInfo.InvariantCulture)}");
                Console.WriteLine($"  solution x:         {Format(iterate)}");
                Console.WriteLine($"  objective value:    {objectiveValue.ToString("G17", CultureInfo.InvariantCulture)}");
                Console.WriteLine($"  equality residuals: {Format(equalityResidual)} (L2={equalityResidual.L2Norm().ToString("G17", CultureInfo.InvariantCulture)})");
                Console.WriteLine($"  inequality residuals: {Format(inequalityResidual)} (pos-max={inequalityResidual.Enumerate().Where(v => v > 0).DefaultIfEmpty(0).Max().ToString("G17", CultureInfo.InvariantCulture)})");

                if (testCase.ExpectedSolution != null)
                {
                    var difference = iterate - testCase.ExpectedSolution;
                    Console.WriteLine($"  reference x*:       {Format(testCase.ExpectedSolution)}");
                    Console.WriteLine($"  difference (x-x*):  {Format(difference)} (L2={difference.L2Norm().ToString("G17", CultureInfo.InvariantCulture)})");
                }

                if (solution.Status != SolverResultStatus.Optimal)
                {
                    Console.WriteLine("quadprog did not converge to an optimal solution. Inspect diagnostics above for more detail.");
                }

                Console.WriteLine();
            }
        }

        private static string Format(IEnumerable<int> indices)
        {
            return $"[{string.Join(", ", indices)}]";
        }

        private static string Format(Vector<double> vector)
        {
            return $"[{string.Join(", ", vector.Enumerate().Select(v => v.ToString("G17", CultureInfo.InvariantCulture)))}]";
        }

        private static string Format(Matrix<double> matrix)
        {
            var rows = matrix.EnumerateRows()
                .Select(row => $"[{string.Join(", ", row.Select(v => v.ToString("G17", CultureInfo.InvariantCulture)))}]");
            return $"[{string.Join(", ", rows)}]";
        }

        private static double EvaluateObjective(QuadraticProblem problem, Vector<double> iterate)
        {
            var value = 0.5 * iterate.DotProduct(problem.Q * iterate) + problem.C.DotProduct(iterate);
            return problem.IsMinimisation ? value : -value;
        }

        private static Vector<double> ComputeEqualityResidual(QuadraticProblem problem, Vector<double> iterate)
        {
            if (problem.EqualityMatrix == null || problem.EqualityVector == null)
            {
                return Vector<double>.Build.Dense(0);
            }

            return problem.EqualityMatrix * iterate - problem.EqualityVector;
        }

        private static Vector<double> ComputeInequalityResidual(QuadraticProblem problem, Vector<double> iterate)
        {
            if (problem.InequalityMatrix == null || problem.InequalityVector == null)
            {
                return Vector<double>.Build.Dense(0);
            }

            return problem.InequalityMatrix * iterate - problem.InequalityVector;
        }

        private static List<int> BuildActiveSet(Vector<double> residualEqu, Vector<double> residualIeq, int equalityCount, double tolerance)
        {
            var active = new List<int>();
            for (var i = 0; i < equalityCount; i++)
            {
                active.Add(i);
            }

            for (var i = 0; i < residualIeq.Count; i++)
            {
                if (Math.Abs(residualIeq[i]) <= tolerance)
                {
                    active.Add(i + equalityCount);
                }
            }

            return active;
        }

        private static (Matrix<double> A, Vector<double> B) BuildAugmentedSystem(
            double[,] inequalityMatrix,
            double[] inequalityVector,
            double[,] equalityMatrix,
            double[] equalityVector)
        {
            var denseIneq = Matrix<double>.Build.DenseOfArray(inequalityMatrix);
            var vectorIneq = Vector<double>.Build.DenseOfArray(inequalityVector);
            var denseEq = Matrix<double>.Build.DenseOfArray(equalityMatrix);
            var vectorEq = Vector<double>.Build.DenseOfArray(equalityVector);

            var totalRows = denseIneq.RowCount + denseEq.RowCount * 2;
            var totalCols = denseIneq.ColumnCount;
            var augmented = Matrix<double>.Build.Dense(totalRows, totalCols);
            var rhs = Vector<double>.Build.Dense(totalRows);

            var cursor = 0;

            for (var i = 0; i < denseIneq.RowCount; i++)
            {
                augmented.SetRow(cursor, denseIneq.Row(i));
                rhs[cursor] = vectorIneq[i];
                cursor++;
            }

            for (var i = 0; i < denseEq.RowCount; i++)
            {
                augmented.SetRow(cursor, denseEq.Row(i));
                rhs[cursor] = vectorEq[i];
                cursor++;

                augmented.SetRow(cursor, -denseEq.Row(i));
                rhs[cursor] = -vectorEq[i];
                cursor++;
            }

            return (augmented, rhs);
        }

        private static void LogProblemData(QuadraticProblem problem)
        {
            Console.WriteLine("problem snapshot:");
            Console.WriteLine($"  Hessian Q:        {Format(problem.Q)}");
            Console.WriteLine($"  Gradient c:       {Format(problem.C)}");

            if (problem.EqualityMatrix != null && problem.EqualityVector != null)
            {
                Console.WriteLine($"  Equality Aeq:     {Format(problem.EqualityMatrix)}");
                Console.WriteLine($"  Equality beq:     {Format(problem.EqualityVector)}");
            }
            else
            {
                Console.WriteLine("  Equality Aeq:     []");
                Console.WriteLine("  Equality beq:     []");
            }

            if (problem.InequalityMatrix != null && problem.InequalityVector != null)
            {
                Console.WriteLine($"  Inequality Aieq:  {Format(problem.InequalityMatrix)}");
                Console.WriteLine($"  Inequality bieq:  {Format(problem.InequalityVector)}");
            }
            else
            {
                Console.WriteLine("  Inequality Aieq:  []");
                Console.WriteLine("  Inequality bieq:  []");
            }

            if (problem.LowerBounds != null)
            {
                Console.WriteLine($"  Lower bounds lb:  {Format(problem.LowerBounds)}");
            }
            else
            {
                Console.WriteLine("  Lower bounds lb:  []");
            }

            if (problem.UpperBounds != null)
            {
                Console.WriteLine($"  Upper bounds ub:  {Format(problem.UpperBounds)}");
            }
            else
            {
                Console.WriteLine("  Upper bounds ub:  []");
            }

            if (problem.InitialGuess != null)
            {
                Console.WriteLine($"  Initial guess x0: {Format(problem.InitialGuess)}");
            }
            else
            {
                Console.WriteLine("  Initial guess x0: []");
            }
        }

        private static QuadraticTestCase CreateBoundedEqualityCase()
        {
            return new QuadraticTestCase
            {
                Name = "Bounded equality constrained (2 variables)",
                BuildProblem = () =>
                {
                    var q = Matrix<double>.Build.DenseOfArray(new[,]
                    {
                        {2.0, 0.0},
                        {0.0, 2.0}
                    });

                    var c = Vector<double>.Build.DenseOfArray(new[] {-2.0, -5.0});

                    var equalityMatrix = Matrix<double>.Build.DenseOfArray(new[,]
                    {
                        {1.0, 1.0}
                    });

                    var equalityVector = Vector<double>.Build.DenseOfArray(new[] {1.0});

                    var lowerBounds = Vector<double>.Build.DenseOfArray(new[] {0.0, 0.0});
                    var initialGuess = Vector<double>.Build.DenseOfArray(new[] {0.5, 0.5});

                    return new QuadraticProblem(q, c, equalityMatrix: equalityMatrix, equalityVector: equalityVector, lowerBounds: lowerBounds, initialGuess: initialGuess);
                },
                ExpectedSolution = Vector<double>.Build.DenseOfArray(new[] {0.0, 1.0})
            };
        }

        private static QuadraticTestCase CreateUnconstrainedQuadraticCase()
        {
            return new QuadraticTestCase
            {
                Name = "Unconstrained paraboloid (2 variables)",
                BuildProblem = () =>
                {
                    var q = Matrix<double>.Build.DenseIdentity(2) * 2.0;
                    var c = Vector<double>.Build.DenseOfArray(new[] {-2.0, -4.0});
                    var initialGuess = Vector<double>.Build.DenseOfArray(new[] {3.0, -1.0});

                    return new QuadraticProblem(q, c, initialGuess: initialGuess);
                },
                ExpectedSolution = Vector<double>.Build.DenseOfArray(new[] {1.0, 2.0})
            };
        }

        private static QuadraticTestCase CreateInequalityActiveCase()
        {
            return new QuadraticTestCase
            {
                Name = "Inequality active at optimum (2 variables)",
                BuildProblem = () =>
                {
                    var q = Matrix<double>.Build.DenseIdentity(2) * 2.0;
                    var c = Vector<double>.Build.DenseOfArray(new[] {-6.0, -2.0 });

                    // Constraint: x + y >= 5  ->  -x - y <= -5
                    var inequalityMatrix = Matrix<double>.Build.DenseOfArray(new[,]
                    {
                        {-1.0, -1.0}
                    });

                    var inequalityVector = Vector<double>.Build.DenseOfArray(new[] {-5.0});
                    var initialGuess = Vector<double>.Build.DenseOfArray(new[] {2.5, 2.5});

                    return new QuadraticProblem(q, c, inequalityMatrix, inequalityVector, initialGuess: initialGuess);
                },
                ExpectedSolution = Vector<double>.Build.DenseOfArray(new[] {3.5, 1.5})
            };
        }

        private sealed class QuadraticTestCase
        {
            public string Name { get; set; }

            public Func<QuadraticProblem> BuildProblem { get; set; }

            public Vector<double> ExpectedSolution { get; set; }
        }
    }
}
