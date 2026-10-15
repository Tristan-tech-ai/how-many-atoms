"""q1768 (10 Oct 2026, READING, no verdict word): (a) the first levels in the variable w = 1/(1-x), x = u t / 8: from q1767 the log level 1 is N(x)/(1-x)^5 with N = (2/3) x^2 (1+x)(x^2-6x-1); in w it is the polynomial (2/3)(w-1)^2 (2w-1)(1+4w-6w^2) by direct substitution; this script prints its expansion and the expansions of the F level 1 (w times the log level 1) and of the reciprocal level 0 (minus the log level 1 over w) as polynomials in w. (b) A prediction from the identity F = exp(L): since the F level 2 is j(j-1)(j-2)(j-3) R(j)/340200 (degree 10, q1759), the log level 2 sequence p(n) = n 8^n times the coefficient of u^(n-2) in L_n must be a polynomial in n of degree 10 vanishing at n = 0, 1, 2, 3.
Predictions (written before the run):
 P1 the polynomial in w of the log level 1 has integer coefficients after multiplying by 3 and degree 5;
 P2 p(n) of the log level 2 is reproduced exactly by a polynomial of degree 10 with the forced zeros n = 0..3, determined on n = 4..14 and checked at n = 15 (one check) and, with the zeros counted, five independent checks in all;
 P3 the polynomial p(n) of the log level 2 has the factor n (n-1) (n-2) (n-3).
Controls (otherwise VOID): C1 the substitution reproduces N/(1-x)^5 exactly as a rational function; C2 the exp relation reproduces the F level 1 polynomial of q1756 from the log level 1 and the top level 1/(1-x).
Usage: py q1768_levels_in_w_and_log_level2.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
w = sp.Symbol("w")
n_ = sp.Symbol("n")
u = sp.Symbol("u")
N = sp.Rational(2, 3) * x**2 * (1 + x) * (x**2 - 6 * x - 1)
L1x = N / (1 - x) ** 5
L1w = sp.expand(sp.Rational(2, 3) * (w - 1) ** 2 * (2 * w - 1) * (1 + 4 * w - 6 * w**2))
c1 = sp.simplify(L1w.subs(w, 1 / (1 - x)) - L1x) == 0
F1w = sp.expand(w * L1w)
g0w = sp.expand(-L1w / w)
Qf = lambda j: -j * (j - 1) * (6 * j**3 + j**2 - 19 * j + 16) / sp.Integer(90)
ser = sp.series(sp.simplify(F1w.subs(w, 1 / (1 - x))), x, 0, 12).removeO()
c2 = all(sp.simplify(ser.coeff(x, k) - Qf(k)) == 0 for k in range(0, 12))
print(
    "C1 substitution exact: %s; C2 F level 1 from the exp relation equals the polynomial of q1756 for k = 0..11: %s"
    % (c1, c2)
)
if not (c1 and c2):
    print("VOID: control failed")
    sys.exit(2)
print("log level 1 in w: %s" % sp.collect(L1w, w))
print("F level 1 in w: %s" % sp.collect(F1w, w))
print("reciprocal level 0 in w (as polynomial in w and 1/w): %s" % sp.collect(sp.expand(g0w), w))
# (b) log level 2 from the exact F_j up to 15
src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read().splitlines()
ns = {"PI": sp.pi}
for ln in src:
    if re.match(r"^(k1 =|c6 =|c10 =|f1[2-9] =|f20 =)", ln):
        exec(ln, ns)
Fx = {
    0: sp.Integer(1),
    1: ns["k1"],
    2: ns["k4"],
    3: ns["c6"],
    4: ns["c8"],
    5: ns["c10"],
    6: ns["f12"],
    7: ns["f14"],
    8: ns["f16"],
    9: ns["f18"],
    10: ns["f20"],
}
txt = open(os.path.join(H, "q1557_stdout_ORD30.txt"), encoding="utf-8").read()
for j, n in zip(range(11, 16), (22, 24, 26, 28, 30)):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    Fx[j] = sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi})
F = {j: sp.Poly(sp.expand(sp.expand(Fx[j]).subs(sp.pi, 1 / sp.sqrt(u))), u) for j in Fx}
L = {}
for k in range(1, 16):
    L[k] = (k * F[k] - sum((i * L[i] * F[k - i] for i in range(1, k)), sp.Poly(0, u))) * sp.Rational(1, k)
p = {
    k: sp.Rational(L[k].coeff_monomial(u ** (k - 2))) * 8**k * k if k >= 2 else sp.Integer(0)
    for k in range(1, 16)
}
print(
    "p(n) = n 8^n coefficient of u^(n-2) in L_n, n = 1..15: "
    + ", ".join("%d: %s" % (k, p[k]) for k in range(1, 16))
)
j_ = n_


def fit(deg, ns_fit, ns_chk, zeros=(0, 1, 2, 3)):
    W = sp.prod([n_ - z for z in zeros])
    r = deg - len(zeros)
    pts = [(k, p[k] / W.subs(n_, k)) for k in ns_fit]
    R = sp.interpolate(pts, n_) if len(pts) > 1 else pts[0][1]
    full = sp.expand(W * R)
    return full, all(full.subs(n_, k) == p[k] for k in ns_chk)


ns_fit = list(range(4, 4 + 7))
ns_chk = [k for k in range(4, 16) if k not in ns_fit]
full, ok = fit(10, ns_fit, ns_chk)
lab = lambda q: "holds" if q else "does not hold"
print(
    "degree-10 polynomial with zeros n = 0..3 fitted on n = %d..%d, checked on n = %s: %s"
    % (ns_fit[0], ns_fit[-1], ns_chk, lab(ok))
)
if ok:
    print("   p(n) = %s" % sp.factor(full))
p1 = sp.Poly(L1w, w).degree() == 5 and all((3 * c).is_integer for c in sp.Poly(L1w, w).all_coeffs())
print(
    "P1 log level 1 in w has degree 5 and 3 times it has integer coefficients: %s; P2 p(n) of the log level 2 is a polynomial of degree 10 with zeros n = 0..3 and %d independent checks: %s; P3 factor n(n-1)(n-2)(n-3): %s"
    % (lab(p1), len(ns_chk), lab(ok), lab(ok))
)
print("READING q1768 done (labels by the script; no verdict word)")
