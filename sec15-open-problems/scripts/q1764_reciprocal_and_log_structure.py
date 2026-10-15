"""q1764 (10 Oct 2026, FACT TABLE, post hoc, no predictions, no verdict word): the structure by power of u = 1/pi^2 of the series related to the bulk coefficients F_j (F_0 = 1, F_1..F_15 exact): the reciprocal coefficients g_n (g = 1/F as a series in t, g_n = -sum_(j>=1) F_j g_(n-j), the density law of births_law.py) and the logarithm coefficients L_n (ln F(t) = sum L_n t^n, n L_n = n F_n - sum_(k=1)^(n-1) k L_k F_(n-k)). For each n the powers of u present, the top power, and the smallest degree d of a polynomial in n reproducing the coefficient of u^(top-k) times 8^n (times n for L) with at least two independent checks, as in q1758.
Control: the same code on F_j itself reproduces the polynomial of q1756 for k = 1 (degree 5).
Usage: py q1764_reciprocal_and_log_structure.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
n_ = sp.Symbol("n")
u = sp.Symbol("u")
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


def to_u(e):
    return sp.expand(sp.expand(e).subs(sp.pi, 1 / sp.sqrt(u)))  # u = 1/pi^2: only even powers occur


F = {j: sp.Poly(to_u(Fx[j]), u) for j in Fx}
g = {0: sp.Poly(1, u)}
for n in range(1, 16):
    g[n] = -sum((F[j] * g[n - j] for j in range(1, n + 1)), sp.Poly(0, u))
L = {}
for n in range(1, 16):
    L[n] = (n * F[n] - sum((k * L[k] * F[n - k] for k in range(1, n)), sp.Poly(0, u))) * sp.Rational(1, n)


def powers(P):
    return sorted(m[0] for m in P.monoms()) if False else sorted(mm[0] for mm in P.monoms())


def coef(P, p):
    return sp.Rational(P.coeff_monomial(u**p)) if p >= 0 else sp.Integer(0)


def min_degree(seq, js, need=2):
    for d in range(0, len(js) - need):
        pts = [(j, seq[j]) for j in js[: d + 1]]
        poly = sp.expand(sp.interpolate(pts, n_)) if d > 0 else sp.Integer(seq[js[0]])
        if all(poly.subs(n_, j) == seq[j] for j in js[d + 1 :]):
            return d, poly, len(js) - d - 1
    return None, None, 0


d0, p0, c0 = min_degree({j: coef(F[j], j - 1) * 8**j for j in range(2, 16)}, list(range(2, 16)))
ctl = d0 == 5 and sp.simplify(p0 - (-n_ * (n_ - 1) * (6 * n_**3 + n_**2 - 19 * n_ + 16) / 90)) == 0
print("CONTROL F_j, level k = 1: degree %s, equals the polynomial of q1756: %s" % (d0, ctl))
if not ctl:
    print("VOID: control failed")
    sys.exit(2)
for name, S, scale in (("g", g, lambda n: 8**n), ("L", L, lambda n: 8**n * n)):
    print(
        "== series %s: highest power of u present in %s_n for n = 1..15: " % (name, name)
        + " ".join("%d:%d" % (n, S[n].degree()) for n in range(1, 16))
        + "; lowest: "
        + " ".join("%d:%d" % (n, powers(S[n])[0]) for n in range(1, 16))
    )
    top = {n: S[n].degree() for n in range(1, 16)}
    print(
        "   coefficient of the top power times scale(n): "
        + " ".join("%d:%s" % (n, coef(S[n], top[n]) * scale(n)) for n in range(1, 16))
    )
    for k in (0, 1, 2):
        # level k below n-1 (g) or n (L): use p = n - 1 - k for g, n - k for L
        off = 1 if name == "g" else 0
        js = [n for n in range(2, 16) if n - off - k >= (powers(S[n])[0] if S[n] != 0 else 0)]
        seq = {n: coef(S[n], n - off - k) * scale(n) for n in js}
        d, poly, nchk = min_degree(seq, js)
        print(
            "   level k = %d (power u^(n-%d)): %s"
            % (
                k,
                off + k,
                (
                    ("polynomial in n of degree %d with %d checks: %s" % (d, nchk, sp.factor(poly)))
                    if d is not None
                    else "no polynomial of degree <= %d with two checks (points n = %d..15)"
                    % (len(js) - 3, js[0])
                ),
            )
        )
print("TABLE q1764 done (facts; no labels)")
