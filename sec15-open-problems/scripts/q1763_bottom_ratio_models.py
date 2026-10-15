"""q1763 (10 Oct 2026, READING, no verdict word): exact models for the bottom ratios r(j) = b(j+2)/b(j), j = 6..13, where b(j) is the coefficient of the lowest power u^ceil(j/2) of the bulk coefficient F_j (exact values from births_law.py and q1557_stdout_ORD30.txt, as in q1760).
NOT BLIND: the ratios (0.46472, 0.68580, 0.74913, 1.02112, 1.08819, 1.40260, 1.48322, 1.84052) were printed in q1760. Predictions (written at 13:56):
 P1 a single rational function of degree (2, 2) determined exactly on five of the eight ratios reproduces the other three exactly;
 P2 a parity-split model r(j) = a0 + a1 j + (-1)^j (b0 + b1 j) determined on four ratios reproduces the other four exactly;
 P3 whichever model holds, its predictions for r(14) and r(15) are written to a registration file before F_16 and F_17 exist (evaluated only if P1 or P2 holds).
Controls (otherwise VOID): C1 a synthetic (2, 2) law is reproduced by the single function and not by the parity model; C2 a synthetic parity-split law is reproduced by the parity model and not by the single function.
Usage: py q1763_bottom_ratio_models.py (from small_gap_q206/; seconds)."""

import os, re, sys, hashlib, datetime
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
n_ = sp.Symbol("n")


def coeffs(expr):
    return {m[0]: sp.Rational(c) for m, c in sp.Poly(sp.expand(expr.subs(sp.pi, 1 / x)), x).terms()}


src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read().splitlines()
ns = {"PI": sp.pi}
for ln in src:
    if re.match(r"^(k1 =|c6 =|c10 =|f1[2-9] =|f20 =)", ln):
        exec(ln, ns)
F = {
    j: f
    for j, f in zip(
        range(2, 11),
        [ns["k4"], ns["c6"], ns["c8"], ns["c10"], ns["f12"], ns["f14"], ns["f16"], ns["f18"], ns["f20"]],
    )
}
txt = open(os.path.join(H, "q1557_stdout_ORD30.txt"), encoding="utf-8").read()
for j, n in zip(range(11, 16), (22, 24, 26, 28, 30)):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    F[j] = sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi})
b = {}
for j in range(2, 16):
    cf = coeffs(F[j])
    b[j] = cf[min(cf)]
R = {j: b[j + 2] / b[j] for j in range(2, 14)}
JS = list(range(6, 14))


def single(r, fit, chk):
    k = 3
    rows = [[r[m] * m**i for i in range(k)] + [-(m**i) for i in range(k)] for m in fit]
    nsp = sp.Matrix(rows).nullspace()
    if len(nsp) != 1:
        return None
    v = nsp[0]
    f = sp.simplify(sum(v[k + i] * n_**i for i in range(k)) / sum(v[i] * n_**i for i in range(k)))
    return all(sp.simplify(f.subs(n_, m) - r[m]) == 0 for m in chk), f


def parity(r, fit, chk):
    a0, a1, b0, b1 = sp.symbols("a0 a1 b0 b1")
    eqs = [a0 + a1 * m + (-1) ** m * (b0 + b1 * m) - r[m] for m in fit]
    sol = sp.solve(eqs, [a0, a1, b0, b1], dict=True)
    if len(sol) != 1:
        return None
    s = sol[0]
    g = lambda m: (s[a0] + s[a1] * m + (-1) ** m * (s[b0] + s[b1] * m))
    return all(sp.simplify(g(m) - r[m]) == 0 for m in chk), s


def both(r):
    fa = [JS[i] for i in (0, 2, 4, 5, 7)]
    ca = [m for m in JS if m not in fa]
    fb = [JS[i] for i in (0, 1, 2, 3)]
    cb = [m for m in JS if m not in fb]
    return single(r, fa, ca), parity(r, fb, cb)


syn1 = {m: sp.Rational((2 * m + 1) * (m + 3), (m + 2) * (3 * m - 1)) for m in JS}
syn2 = {m: sp.Rational(m, 3) + (-1) ** m * sp.Rational(m + 1, 7) for m in JS}
a1, p1 = both(syn1)
a2, p2 = both(syn2)
c1 = a1 is not None and a1[0] is True and (p1 is None or p1[0] is False)
c2 = p2 is not None and p2[0] is True and (a2 is None or a2[0] is False)
print(
    "C1 synthetic (2, 2) law: single function reproduces, parity model does not: %s; C2 synthetic parity law: parity model reproduces, single function does not: %s"
    % (c1, c2)
)
if not (c1 and c2):
    print("VOID: control failed")
    sys.exit(2)
sa, pa = both(R)
print("ratios r(j), j = 6..13: " + "; ".join("%d: %s (%.5f)" % (m, R[m], float(R[m])) for m in JS))
print(
    "single (2, 2) function on j = 6, 8, 10, 11, 13: %s"
    % (("reproduces j = 7, 9, 12 exactly: %s" % sa[0]) if sa is not None else "no unique fit")
)
print(
    "parity model on j = 6..9: %s"
    % (
        (
            "reproduces j = 10..13 exactly: %s; coefficients %s"
            % (pa[0], {str(k): v for k, v in pa[1].items()})
        )
        if pa is not None
        else "no unique solution"
    )
)
lab = lambda q: "holds" if q else "does not hold"
q1 = sa is not None and sa[0] is True
q2 = pa is not None and pa[0] is True
print(
    "P1 single rational function of degree (2, 2) reproduces three held-out ratios: %s; P2 parity-split linear model reproduces four held-out ratios: %s; P3: %s"
    % (
        lab(q1),
        lab(q2),
        "predictions to be registered" if (q1 or q2) else "not evaluated (neither model holds)",
    )
)
print("READING q1763 done (labels by the script; no verdict word)")
