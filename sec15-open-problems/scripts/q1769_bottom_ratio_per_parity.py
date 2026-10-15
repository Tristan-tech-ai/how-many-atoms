"""q1769 (10 Oct 2026, READING, no verdict word): is the bottom ratio b(j+2)/b(j) of the bulk coefficients, for each parity of j separately, an exact polynomial of degree 2 in j (Gamma-function type growth), or an exact rational function of degree (2, 1)? b(j) is the coefficient of the lowest power u^ceil(j/2) of F_j (exact values as in q1760 and q1763).
NOT BLIND: the ratios were printed in q1760 and q1763. Predictions (written at 14:01):
 P1 for each parity separately the ratios for j = 2..12 (even) and j = 3..13 (odd) lie exactly on a polynomial of degree 2 in j determined on three of them, with the other three as checks;
 P2 for the parity where P1 fails, a rational function of degree (2, 1) (4 parameters, 2 checks) reproduces them;
 P3 if neither holds, the prime factorisations of the numerators and denominators of b(j) for j = 6..15 are listed and the primes above 31 are counted.
Controls (otherwise VOID): C1 a synthetic Gamma-type sequence with the polynomial ratio (j+1)(j+4)/7 is reproduced by P1; C2 the same with an added term 1/(j+3) is not reproduced by P1.
Usage: py q1769_bottom_ratio_per_parity.py (from small_gap_q206/; seconds)."""

import os, re, sys
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


def poly_fit(r, js):
    pts = [(m, r[m]) for m in js[:3]]
    P = sp.expand(sp.interpolate(pts, n_))
    return P, all(P.subs(n_, m) == r[m] for m in js[3:])


def rat_fit(r, js, dn=1, dd=2):
    # r = (a0 + a1 n + a2 n^2) / (1 + d1 n) with 4 parameters
    a0, a1, a2, d1 = sp.symbols("a0 a1 a2 d1")
    eqs = [a0 + a1 * m + a2 * m**2 - r[m] * (1 + d1 * m) for m in js[:4]]
    sol = sp.solve(eqs, [a0, a1, a2, d1], dict=True)
    if len(sol) != 1:
        return None, False
    s = sol[0]
    f = lambda m: (s[a0] + s[a1] * m + s[a2] * m**2) / (1 + s[d1] * m)
    return s, all(sp.simplify(f(m) - r[m]) == 0 for m in js[4:])


syn1 = {j: sp.Integer(1) for j in range(2, 16)}
syn2 = dict(syn1)
for j in range(2, 14):
    syn1[j + 2] = syn1[j] * sp.Rational((j + 1) * (j + 4), 7)
    syn2[j + 2] = syn2[j] * (sp.Rational((j + 1) * (j + 4), 7) + sp.Rational(1, j + 3))


def ratios(seq, par):
    js = [j for j in range(2, 14) if j % 2 == par]
    return {j: seq[j + 2] / seq[j] for j in js}, js


c1 = all(poly_fit(*ratios(syn1, p))[1] for p in (0, 1))
c2 = all(not poly_fit(*ratios(syn2, p))[1] for p in (0, 1))
print(
    "C1 synthetic Gamma-type ratio reproduced by the quadratic fit for both parities: %s; C2 synthetic with an added term not reproduced: %s"
    % (c1, c2)
)
if not (c1 and c2):
    print("VOID: control failed")
    sys.exit(2)
res = {}
for par in (0, 1):
    r, js = ratios(b, par)
    P, ok = poly_fit(r, js)
    res[par] = (r, js, P, ok)
    print(
        "parity %s: ratios %s"
        % ("even" if par == 0 else "odd", "; ".join("%d: %.5f" % (m, float(r[m])) for m in js))
    )
    print(
        "   quadratic through the first three ratios: %s ; reproduces the other three exactly: %s"
        % (sp.nsimplify(P), ok)
    )
for par in (0, 1):
    r, js, P, ok = res[par]
    if not ok:
        s, ok2 = rat_fit(r, js)
        res[par] = res[par] + (ok2,)
        print(
            "parity %s: (2, 1) rational fit on the first four, checks on the last two: %s"
            % ("even" if par == 0 else "odd", ("reproduces: %s" % ok2) if s is not None else "no unique fit")
        )
    else:
        res[par] = res[par] + (None,)
lab = lambda q: "holds" if q else "does not hold"
p1 = all(res[p][3] for p in (0, 1))
fails = [p for p in (0, 1) if not res[p][3]]
p2 = bool(fails) and all(res[p][4] for p in fails)
print(
    "P1 exact quadratic ratio for both parities: %s; P2 (2, 1) rational reproduces the parity that fails P1: %s"
    % (lab(p1), (lab(p2) if fails else "not applicable (P1 holds)"))
)
if not p1 and not p2:
    big = {}
    for j in range(6, 16):
        f = sp.factorint(abs(b[j].p))
        g = sp.factorint(b[j].q)
        big[j] = (sorted(k for k in f if k > 31), sorted(k for k in g if k > 31))
    print(
        "P3 primes above 31 in the numerator and denominator of b(j), j = 6..15: "
        + "; ".join("%d: %s / %s" % (j, big[j][0], big[j][1]) for j in big)
    )
    print(
        "   count of primes above 31: numerators %d, denominators %d"
        % (sum(len(big[j][0]) for j in big), sum(len(big[j][1]) for j in big))
    )
print("READING q1769 done (labels by the script; no verdict word)")
