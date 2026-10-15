"""q1749 (10 Oct 2026, READING, no verdict word): structure of the exact centre-series terms e_n as polynomials in pi^2 with rational coefficients (centre_scripts/q1634_out_10.json, keys e2..e10). Do the top-power coefficients c_top(n) (power pi^(2n-2)) and the constant coefficients c_0(n) follow a ratio law that is a rational function of n?
NOT BLIND: the closed forms of e_4 and e_5 were seen. Predictions (written at 13:36):
 P1 the ratios q_n = c_top(n+1)/c_top(n) for n = 3..9 are reproduced exactly (rational arithmetic) by a rational function (a n^2 + b n + c)/(d n^2 + e n + f) determined on n = 3..7, at n = 8 and 9;
 P2 the same test on the pi^0 coefficients c_0(n) does NOT reproduce the ratios at n = 8 and 9 (a control that can fail);
 P3 (read off after P1) the top coefficient has a closed product form; this item is only evaluated when P1 holds and is then reported as the factorised numerator and denominator of the fitted function.
Controls (otherwise VOID): C1 a synthetic ratio sequence q_n = (2n+1)(n+3)/((n+2)(3n-1)) is reproduced exactly at n = 8, 9 by the same code; C2 a synthetic sequence q_n = (n+1)/(n+2) (1 + 2^(-n)) is not.
Usage: py q1749_pi2_structure.py (from small_gap_q206/; seconds)."""

import json, os, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
pi = sp.pi
x = sp.Symbol("x")
D = json.load(open(os.path.join(H, "..", "centre_scripts", "q1634_out_10.json")))
POLY = {}
for n in range(2, 11):
    ex = sp.expand(sp.sympify(D["e%d" % n].replace("pi", "PI"), locals={"PI": x}))  # x stands for pi
    P = sp.Poly(ex, x)
    POLY[n] = {m[0]: c for m, c in P.terms()}
print("powers of pi present: " + "; ".join("e_%d: %s" % (n, sorted(POLY[n])) for n in range(2, 11)))
top = {n: (max(POLY[n]), POLY[n][max(POLY[n])]) for n in range(2, 11)}
print(
    "top power of e_n: "
    + ", ".join("%d: pi^%d" % (n, top[n][0]) for n in range(2, 11))
    + "; expected 2n-2: "
    + str(all(top[n][0] == 2 * n - 2 for n in range(3, 11)))
)


def ratios(c, ns):
    return {n: sp.Rational(c[n + 1]) / sp.Rational(c[n]) for n in ns}


def fit_check(q, fit_ns, check_ns, dn=2):
    n_ = sp.Symbol("n")
    k = dn + 1
    rows = [
        [q[m] * m**i for i in range(k)] + [-(m**i) for i in range(k)] for m in fit_ns
    ]  # q (d-part) - (a-part) = 0
    ns_ = sp.Matrix(rows).nullspace()
    if len(ns_) != 1:
        return None, "nullspace dimension %d" % len(ns_)
    v = ns_[0]
    den = sum(v[i] * n_**i for i in range(k))
    num = sum(v[k + i] * n_**i for i in range(k))
    f = sp.simplify(num / den)
    ok = all(sp.simplify(f.subs(n_, m) - q[m]) == 0 for m in check_ns)
    return (ok, sp.factor(f)), "ok"


def run(c, label):
    q = ratios(c, range(3, 10))
    res, msg = fit_check(q, range(3, 8), (8, 9))
    return q, res, msg


ct = {n: top[n][1] for n in range(3, 11)}
c0 = {n: POLY[n].get(0, 0) for n in range(3, 11)}
syn1 = {3: sp.Integer(1)}
syn2 = {3: sp.Integer(1)}
for n in range(3, 10):
    syn1[n + 1] = syn1[n] * sp.Rational((2 * n + 1) * (n + 3), (n + 2) * (3 * n - 1))
    syn2[n + 1] = syn2[n] * sp.Rational(n + 1, n + 2) * (1 + sp.Rational(1, 2**n))
r1, f1, m1 = run(syn1, "C1")
r2, f2, m2 = run(syn2, "C2")
c1 = f1 is not None and f1[0] is True
c2 = f2 is None or f2[0] is False
print(
    "C1 synthetic rational ratio law reproduced exactly at n = 8, 9: %s; C2 synthetic non-rational ratio not reproduced: %s"
    % (c1, c2)
)
if not (c1 and c2):
    print("VOID: control failed")
    sys.exit(2)
qt, ft, mt = run(ct, "top")
q0, f0, m0 = run(c0, "const")
print(
    "top coefficients c_top(n), n = 3..10 (numerators/denominators): "
    + "; ".join("%d: %s" % (n, ct[n]) for n in range(3, 11))
)
print(
    "ratios q_n = c_top(n+1)/c_top(n), n = 3..9: "
    + "; ".join("%d: %s (%.6f)" % (n, qt[n], float(qt[n])) for n in range(3, 10))
)
print(
    "top: fit on n = 3..7 -> %s"
    % (
        ("reproduces n = 8, 9: %s; function %s" % (ft[0], ft[1]))
        if ft is not None
        else msg_ if False else ("no unique fit (%s)" % mt)
    )
)
print("constant coefficients c_0(n), n = 3..10: " + "; ".join("%d: %s" % (n, c0[n]) for n in range(3, 11)))
print("ratios of c_0: " + "; ".join("%d: %.6f" % (n, float(q0[n])) for n in range(3, 10)))
print(
    "constant: fit on n = 3..7 -> %s"
    % (
        ("reproduces n = 8, 9: %s; function %s" % (f0[0], f0[1]))
        if f0 is not None
        else ("no unique fit (%s)" % m0)
    )
)
lab = lambda q: "holds" if q else "does not hold"
p1 = ft is not None and ft[0] is True
p2 = (f0 is None) or (f0[0] is False)
print(
    "P1 top-coefficient ratios reproduced exactly by a rational function of degree (2, 2): %s; P2 constant-coefficient ratios not reproduced: %s; P3 closed form: %s"
    % (lab(p1), lab(p2), ("factorised function %s" % ft[1]) if p1 else "not evaluated (P1 does not hold)")
)
print("READING q1749 done (labels by the script; no verdict word)")
