"""q1770 (10 Oct 2026, READING, no verdict word): the log level 2 as a polynomial in w = 1/(1-x), x = u t / 8. The sequence p(n) = n 8^n times the coefficient of u^(n-2) in L_n is the degree-10 polynomial of q1768, p(n) = n (n-1) (n-2) (n-3) (115 n^6 - 1425 n^5 + 5137 n^4 + 945 n^3 - 40676 n^2 + 70392 n - 26964) / 34020. The coefficient of x^n in the level is p(n)/n, so the level is the generating function G(x) = sum_(n>=1) p(n)/n x^n = N(x)/(1-x)^10, and in w it is the polynomial N(1 - 1/w) w^10.
NOT BLIND: p(n) and the w-form of the level 1 (q1768) are known. Predictions (written at 14:01):
 P1 L_2(w) has the factor (w-1)^2 or a higher power of (w-1);
 P2 the leading coefficient of L_2(w) in w^10 is a rational multiple of the square of the leading coefficient -8 of L_1(w) in w^5, and the multiple is printed;
 P3 L_2(w) and L_1(w) have a common root other than w = 1.
Controls (otherwise VOID): C1 the series of G(x) reproduces p(n)/n for n = 1..25; C2 the same code on the level 1, p1(n)/n = -(n-1)(n+1)(3n^2-8n+6)/9, gives the w-polynomial of q1768.
Usage: py q1770_log_level2_in_w.py (from small_gap_q206/; seconds)."""

import sys
import sympy as sp

x = sp.Symbol("x")
n = sp.Symbol("n")
w = sp.Symbol("w")
ORDER = 40


def to_w(pn, n0, d):
    ser = sum(pn.subs(n, k) * x**k for k in range(n0, ORDER))
    prod = sp.expand(ser * (1 - x) ** (d + 1))
    N = sum(prod.coeff(x, k) * x**k for k in range(0, d + 3))
    rest = all(prod.coeff(x, k) == 0 for k in range(d + 3, ORDER))
    Lw = sp.expand(sp.simplify(N.subs(x, 1 - 1 / w) * w ** (d + 1)))
    return N, rest, Lw


p1n = -(n - 1) * (n + 1) * (3 * n**2 - 8 * n + 6) / sp.Integer(9)
N1, ok1, L1w = to_w(sp.expand(p1n), 1, 4)
ref = -8 * w**5 + sp.Rational(76, 3) * w**4 - 28 * w**3 + sp.Rational(34, 3) * w**2 - sp.Rational(2, 3)
c2 = ok1 and sp.expand(L1w - ref) == 0
p2 = (
    n
    * (n - 1)
    * (n - 2)
    * (n - 3)
    * (115 * n**6 - 1425 * n**5 + 5137 * n**4 + 945 * n**3 - 40676 * n**2 + 70392 * n - 26964)
    / sp.Integer(34020)
)
q2n = sp.cancel(p2 / n)
N2, ok2, L2w = to_w(sp.expand(q2n), 1, 9)
ser = sp.series(N2 / (1 - x) ** 10, x, 0, 26).removeO()
c1 = ok2 and all(ser.coeff(x, k) == q2n.subs(n, k) for k in range(1, 26))
print(
    "C1 series of G reproduces p(n)/n for n = 1..25: %s; C2 level 1 gives the w-polynomial of q1768: %s"
    % (c1, c2)
)
if not (c1 and c2):
    print("VOID: control failed")
    sys.exit(2)
print("L_2(w) = %s" % sp.collect(L2w, w))
print("factored: %s" % sp.factor(L2w))
lead = sp.Poly(L2w, w).LC()
print("degree in w: %d; leading coefficient %s; divided by (-8)^2: %s" % (sp.degree(L2w, w), lead, lead / 64))
mult = sp.roots(sp.Poly(L2w, w)).get(1, 0) if True else 0
print(
    "multiplicity of w = 1 in L_2: %s; in L_1: %s"
    % (
        sp.Poly(sp.factor_list(L2w)[1] and L2w, w).degree() and sp.roots(sp.Poly(L2w, w)).get(1, 0),
        sp.roots(sp.Poly(L1w, w)).get(1, 0),
    )
)
r1 = sp.Poly(L1w, w)
r2 = sp.Poly(L2w, w)
g = sp.gcd(r1, r2)
print("gcd of L_1 and L_2 as polynomials in w: %s" % sp.factor(g.as_expr()))
lab = lambda q: "holds" if q else "does not hold"
m1 = sp.roots(r2).get(1, 0)
gq = sp.cancel(g.as_expr() / (w - 1) ** min(sp.roots(r1).get(1, 0), m1)) if m1 else g.as_expr()
p1 = m1 >= 2
p2 = lead / 64 == sp.nsimplify(lead / 64)
p3 = sp.degree(gq, w) >= 1
print(
    "P1 L_2 has (w-1)^2 or higher: %s (multiplicity %s); P2 the leading coefficient over (-8)^2 is the rational number %s: %s; P3 a common root other than w = 1: %s (cofactor %s)"
    % (lab(p1), m1, lead / 64, lab(p2), lab(p3), sp.factor(gq))
)
print("READING q1770 done (labels by the script; no verdict word)")
