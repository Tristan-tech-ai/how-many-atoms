"""q1767 (10 Oct 2026, READING, no verdict word): rational generating functions of the first levels of the series of q1764 and q1756. With x = u t / 8, a level whose coefficient is a polynomial p(n) of degree d (for n from n0 on) has the generating function N(x) / (1-x)^(d+1) with N a polynomial. The three levels: the logarithm level 1, p(n) = -(n-1)(n+1)(3n^2-8n+6)/9 (the coefficient of u^(n-1) times 8^n, with the factor n divided out, summed from n = 1); the F level 1, p(j) = -j(j-1)(6j^3+j^2-19j+16)/90 (from j = 0); the reciprocal level 0, p(n) = 2(2n^3-7n^2+7n-1)/3 (from n = 2).
NOT BLIND: the polynomials of q1764 are known. Predictions (written at 14:01):
 P1 for the logarithm level 1, N has degree at most 4 and 9 N has integer coefficients;
 P2 N is divisible by x^2 for the logarithm level 1 (no terms n = 0, 1);
 P3 N(1) equals d! times the leading coefficient of p (d = 4 for the logarithm level).
Controls (otherwise VOID): C1 the sequence n^2 for n >= 1 gives N = x (1 + x) with d = 2, found by the same code; C2 the code reproduces the series coefficients of p from N / (1-x)^(d+1) for all three levels to order 25.
Usage: py q1767_levels_generating_functions.py (from small_gap_q206/; seconds)."""

import sys
import sympy as sp

x = sp.Symbol("x")
n = sp.Symbol("n")
ORDER = 30


def gen(p, n0, d):
    ser = sum(p.subs(n, k) * x**k for k in range(n0, ORDER))
    prod = sp.expand(ser * (1 - x) ** (d + 1))
    N = sum(prod.coeff(x, k) * x**k for k in range(0, d + 3))
    rest = [prod.coeff(x, k) for k in range(d + 3, ORDER)]
    return sp.expand(N), all(r == 0 for r in rest)


ctl1 = gen(n**2, 1, 2)
c1 = ctl1[1] and sp.expand(ctl1[0] - x * (1 + x)) == 0
print("C1 sequence n^2: N = %s, remaining coefficients zero: %s, equals x(1+x): %s" % (ctl1[0], ctl1[1], c1))
if not c1:
    print("VOID: control failed")
    sys.exit(2)
levels = {
    "log level 1": (-(n - 1) * (n + 1) * (3 * n**2 - 8 * n + 6) / 9, 1, 4),
    "F level 1": (-n * (n - 1) * (6 * n**3 + n**2 - 19 * n + 16) / 90, 0, 5),
    "reciprocal level 0": (2 * (2 * n**3 - 7 * n**2 + 7 * n - 1) / 3, 2, 3),
}
res = {}
for name, (p, n0, d) in levels.items():
    N, ok = gen(sp.expand(p), n0, d)
    res[name] = (N, ok, d, p)
    chk = all(
        sp.series(N / (1 - x) ** (d + 1), x, 0, 26).removeO().coeff(x, k) == p.subs(n, k)
        for k in range(n0, 26)
    )
    print(
        "%s: p has degree %d; N(x) = %s ; factored %s ; series check to order 25: %s ; remaining coefficients zero: %s ; N(1) = %s"
        % (name, d, N, sp.factor(N), chk, ok, N.subs(x, 1))
    )
    if not (chk and ok):
        print("VOID: control C2 failed")
        sys.exit(2)
N, ok, d, p = res["log level 1"]
lab = lambda q: "holds" if q else "does not hold"
deg = sp.degree(N, x)
p1 = deg <= 4 and all((9 * c).is_integer for c in sp.Poly(N, x).all_coeffs())
p2 = sp.rem(N, x**2, x) == 0
p3 = N.subs(x, 1) == sp.factorial(d) * sp.Poly(sp.expand(p), n).LC()
print("log level 1: degree of N = %d" % deg)
print(
    "P1 N has degree at most 4 and 9 N has integer coefficients: %s; P2 N divisible by x^2: %s; P3 N(1) = 4! times the leading coefficient: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1767 done (labels by the script; no verdict word)")
