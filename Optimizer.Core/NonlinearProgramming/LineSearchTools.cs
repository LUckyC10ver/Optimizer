using System;

namespace Optimizer.Core.NonlinearProgramming
{
    /// <summary>
    /// Helper routines that implement classic one-dimensional search strategies used by
    /// the historic optimisation toolbox. They closely mirror the original algorithms
    /// from <c>OPTtools.h</c> so that ported code can rely on the same numerical
    /// behaviour.
    /// </summary>
    internal static class LineSearchTools
    {
        private const double GoldenRatio = 1.618034;        // 2 / (sqrt(5) - 1)
        private const double Tiny = 1e-20;
        private const double GrowthLimit = 100.0;
        private const double ConjugateGolden = 0.3819660;   // 1 - 1/GoldenRatio
        private const double Zeps = 1e-10;

        /// <summary>
        /// Attempts to bracket a minimum of <paramref name="function"/> starting from
        /// the interval [<paramref name="a"/>, <paramref name="cMax"/>]. The method
        /// mirrors <c>OPTbracketMinimum</c> from the legacy implementation.
        /// </summary>
        public static (double a, double b, double c, double fa, double fb, double fc) BracketMinimum(
            double a,
            double b,
            double cMax,
            Func<double, double> function)
        {
            var fa = function(a);
            var fb = function(b);

            if (fb > fa)
            {
                Swap(ref a, ref b);
                Swap(ref fa, ref fb);
            }

            var c = Math.Min(b + GoldenRatio * (b - a), cMax);
            var fc = function(c);

            while (fb > fc)
            {
                var r = (b - a) * (fb - fc);
                var q = (b - c) * (fb - fa);
                var denom = 2.0 * SignedValue(q - r) * Math.Max(Math.Abs(q - r), Tiny);
                var u = b - ((b - c) * q - (b - a) * r) / denom;
                var uLimit = Math.Min(b + GrowthLimit * (c - b), cMax);

                double fu;
                if ((b - u) * (u - c) > 0.0)
                {
                    fu = function(u);
                    if (fu < fc)
                    {
                        a = b;
                        fa = fb;
                        b = u;
                        fb = fu;
                        break;
                    }

                    if (fu > fb)
                    {
                        c = u;
                        fc = fu;
                        break;
                    }

                    u = c + GoldenRatio * (c - b);
                    fu = function(u);
                }
                else if ((c - u) * (u - uLimit) > 0.0)
                {
                    fu = function(u);
                    if (fu < fc)
                    {
                        a = b;
                        fa = fb;
                        b = c;
                        fb = fc;
                        c = u;
                        fc = fu;

                        u = c + GoldenRatio * (c - b);
                        fu = function(u);
                    }
                }
                else if ((u - uLimit) * (uLimit - c) >= 0.0)
                {
                    u = uLimit;
                    fu = function(u);
                }
                else
                {
                    u = c + GoldenRatio * (c - b);
                    fu = function(u);
                }

                a = b;
                fa = fb;
                b = c;
                fb = fc;
                c = u;
                fc = fu;
            }

            return (a, b, c, fa, fb, fc);
        }

        /// <summary>
        /// Performs a golden section search within a bracketing triplet. The routine
        /// returns both the minimiser and its function value.
        /// </summary>
        public static (double value, double position) GoldenSectionSearch(
            double ax,
            double bx,
            double cx,
            Func<double, double> function,
            double tolerance)
        {
            var r = 0.61803399;
            var c = 1.0 - r;

            var x0 = ax;
            var x3 = cx;
            double x1, x2;

            if (Math.Abs(cx - bx) > Math.Abs(bx - ax))
            {
                x1 = bx;
                x2 = bx + c * (cx - bx);
            }
            else
            {
                x2 = bx;
                x1 = bx - c * (bx - ax);
            }

            var f1 = function(x1);
            var f2 = function(x2);

            while (Math.Abs(x3 - x0) > tolerance * (Math.Abs(x1) + Math.Abs(x2)))
            {
                if (f2 < f1)
                {
                    x0 = x1;
                    x1 = x2;
                    x2 = r * x1 + c * x3;
                    f1 = f2;
                    f2 = function(x2);
                }
                else
                {
                    x3 = x2;
                    x2 = x1;
                    x1 = r * x2 + c * x0;
                    f2 = f1;
                    f1 = function(x1);
                }
            }

            if (f1 < f2)
            {
                return (f1, x1);
            }

            return (f2, x2);
        }

        /// <summary>
        /// Minimises the supplied function using Brent's method. The implementation is a
        /// direct translation of the original toolbox routine.
        /// </summary>
        public static (double value, double position) BrentSearch(
            double ax,
            double bx,
            double cx,
            Func<double, double> function,
            double tolerance,
            int maxIterations = 100)
        {
            var a = Math.Min(ax, cx);
            var b = Math.Max(ax, cx);
            var x = bx;
            var w = x;
            var v = x;
            var fw = function(w);
            var fv = fw;
            var fx = fw;
            var e = 0.0;
            var d = 0.0;

            for (var iter = 0; iter < maxIterations; iter++)
            {
                var xm = 0.5 * (a + b);
                var tol1 = tolerance * Math.Abs(x) + Zeps;
                var tol2 = 2.0 * tol1;

                if (Math.Abs(x - xm) <= tol2 - 0.5 * (b - a))
                {
                    return (fx, x);
                }

                if (Math.Abs(e) > tol1)
                {
                    var r = (x - w) * (fx - fv);
                    var q = (x - v) * (fx - fw);
                    var p = (x - v) * q - (x - w) * r;
                    q = 2.0 * (q - r);
                    if (q > 0.0)
                    {
                        p = -p;
                    }

                    q = Math.Abs(q);
                    var etemp = e;
                    e = d;

                    if (Math.Abs(p) >= Math.Abs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x))
                    {
                        e = x >= xm ? a - x : b - x;
                        d = ConjugateGolden * e;
                    }
                    else
                    {
                        d = p / q;
                        var u = x + d;
                        if (u - a < tol2 || b - u < tol2)
                        {
                            d = SignedValue(xm - x) * tol1;
                        }
                    }
                }
                else
                {
                    e = x >= xm ? a - x : b - x;
                    d = ConjugateGolden * e;
                }

                var uCandidate = Math.Abs(d) >= tol1 ? x + d : x + SignedValue(d) * tol1;
                var fu = function(uCandidate);

                if (fu <= fx)
                {
                    if (uCandidate >= x)
                    {
                        a = x;
                    }
                    else
                    {
                        b = x;
                    }

                    v = w;
                    fv = fw;
                    w = x;
                    fw = fx;
                    x = uCandidate;
                    fx = fu;
                }
                else
                {
                    if (uCandidate < x)
                    {
                        a = uCandidate;
                    }
                    else
                    {
                        b = uCandidate;
                    }

                    if (fu <= fw || Math.Abs(w - x) < double.Epsilon)
                    {
                        v = w;
                        fv = fw;
                        w = uCandidate;
                        fw = fu;
                    }
                    else if (fu <= fv || Math.Abs(v - x) < double.Epsilon || Math.Abs(v - w) < double.Epsilon)
                    {
                        v = uCandidate;
                        fv = fu;
                    }
                }
            }

            return (fx, x);
        }

        private static void Swap<T>(ref T lhs, ref T rhs)
        {
            var temp = lhs;
            lhs = rhs;
            rhs = temp;
        }

        private static double SignedValue(double value)
        {
            return value >= 0.0 ? 1.0 : -1.0;
        }
    }
}
