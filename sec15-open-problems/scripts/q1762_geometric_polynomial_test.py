"""q1762 (10 Oct 2026, READING, no verdict word; REPAIR of q1761, whose first run ended VOID and is kept as q1761_out_first_run_void_control.txt): the control C2 of q1761 demanded that Z be far from 1 for the synthetic sum 8^n + 3^n, but Z is already close to 1 at n = 8 for any sequence dominated by one geometric term (and for n!, Z = (n+2)/(n+1)), so Z cannot reject such forms and P1 is a weak band. C2 now demands only that the finite differences of 8^n + 3^n do not vanish at lambda = 8. Predictions and bars are unchanged.
Original docstring of q1761: : do the coefficients d_m(n) of pi^(2n-2-2m) in the exact e_n (n = 3..10, m = 0, 1, 2) have the form lambda^n times a polynomial in n? The lambda-free ratio Z_m(n) = d_m(n+2) d_m(n) / d_m(n+1)^2 equals P(n+2) P(n) / P(n+1)^2 for such a form and tends to 1 away from the roots of P. The finite differences of d_m(n) lambda^(-n) are also printed for lambda = 8, 1/8, 16, 4, 2 and 1, with the order at which they vanish exactly on n = 3..10.
NOT BLIND: the ratios of d_m were seen in q1752. Predictions (written at 13:51):
 P1 for m = 1 the values Z_1(n), n = 5..8, lie between 0.8 and 1.2;
 P2 for m = 1 and lambda = 8 the lowest finite-difference order of d_1(n)/8^n that vanishes exactly on n = 3..10 is at most 5;
 P3 for m = 0, Z_0(n) leaves the band 0.5 to 2 for some n in 3..8 (a root of P near the sign change).
Controls (otherwise VOID): C1 a synthetic d(n) = 8^n (n^3 - 4n + 1) / 7 gives Z equal to its closed form to 1e-12 and a vanishing difference order 4 at lambda = 8; C2 a synthetic sum 8^n + 3^n has no vanishing order at lambda = 8 and a Z far from 1.
Usage: py q1761_geometric_polynomial_test.py (from small_gap_q206/; seconds)."""

import json, os, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
D = json.load(open(os.path.join(H, "..", "centre_scripts", "q1634_out_10.json")))
POLY = {}
for n in range(3, 11):
    ex = sp.expand(sp.sympify(D["e%d" % n].replace("pi", "PI"), locals={"PI": x}))
    POLY[n] = {m[0]: sp.Rational(c) for m, c in sp.Poly(ex, x).terms()}


def Z(d, ns):
    return {n: d[n + 2] * d[n] / d[n + 1] ** 2 for n in ns}


def vanishing_order(seq, ns):
    v = [seq[n] for n in ns]
    order = 0
    while len(v) > 1:
        if all(t == 0 for t in v):
            return order
        v = [b - a for a, b in zip(v[:-1], v[1:])]
        order += 1
    return None


def fd_order(d, lam):
    return vanishing_order({n: d[n] / lam**n for n in d}, sorted(d))


P = lambda n: sp.Rational(n**3 - 4 * n + 1, 7)
c1d = {n: sp.Integer(8) ** n * P(n) for n in range(3, 11)}
c2d = {n: sp.Integer(8) ** n + sp.Integer(3) ** n for n in range(3, 11)}
z1 = Z(c1d, range(3, 9))
c1 = all(z1[n] == P(n + 2) * P(n) / P(n + 1) ** 2 for n in z1) and fd_order(c1d, sp.Integer(8)) == 4
z2 = Z(c2d, range(3, 9))
c2 = fd_order(c2d, sp.Integer(8)) is None
print(
    "C1 synthetic 8^n times a cubic: Z closed form and difference order 4 at lambda = 8: %s; C2 synthetic 8^n + 3^n: no vanishing order at lambda = 8: %s"
    % (c1, c2)
)
if not (c1 and c2):
    print("VOID: control failed")
    sys.exit(2)
out = {}
for m in (0, 1, 2):
    d = {n: POLY[n].get(2 * n - 2 - 2 * m, 0) for n in range(3, 11)}
    z = Z(d, range(3, 9))
    out[m] = (d, z)
    print("m = %d: Z(n), n = 3..8: %s" % (m, " ".join("%d: %.4f" % (n, float(z[n])) for n in range(3, 9))))
    print(
        "   vanishing order of the finite differences of d_m(n) lambda^-n on n = 3..10: "
        + ", ".join(
            "lambda=%s: %s" % (lam, fd_order(d, sp.Rational(lam)))
            for lam in ("8", "1/8", "16", "4", "2", "1")
        )
    )
lab = lambda q: "holds" if q else "does not hold"
z1v = [float(out[1][1][n]) for n in range(5, 9)]
o8 = fd_order(out[1][0], sp.Integer(8))
z0 = [float(out[0][1][n]) for n in range(3, 9)]
p1 = all(0.8 <= v <= 1.2 for v in z1v)
p2 = o8 is not None and o8 <= 5
p3 = any(v < 0.5 or v > 2 for v in z0)
print(
    "(P1 is a weak band: Z is close to 1 for any sequence dominated by one geometric term; P2 is the discriminating item)"
)
print(
    "P1 Z_1(5..8) in [0.8, 1.2]: %s (%s); P2 vanishing order of d_1(n)/8^n at most 5: %s (order %s); P3 Z_0 leaves [0.5, 2]: %s"
    % (lab(p1), " ".join("%.4f" % v for v in z1v), lab(p2), o8, lab(p3))
)
print("READING q1762 done (labels by the script; no verdict word)")
