"""q1752 (10 Oct 2026, READING, no verdict word): split of the exact centre-series terms e_n (centre_scripts/q1634_out_10.json, polynomials in pi^2) by the distance m from the top power: d_m(n) = coefficient of pi^(2n-2-2m) in e_n, n = 3..10. Do the ratios d_m(n+1)/d_m(n) for n = 3..9 follow a rational function of degree (2, 2) determined exactly on n = 3..7 and checked at n = 8 and 9?
NOT BLIND: the sequences m = 0 (top) and the constant coefficients were seen in q1749. Predictions (written at 13:44):
 P1 at least one of the sequences m = 1, 2 has its ratios at n = 8 and 9 reproduced exactly by the function determined on n = 3..7;
 P2 the sequence m = 0 (q1749) still does not (the control that can fail, repeated);
 P3 if P1 holds, the fitted function is printed factorised and compared with the sign pattern of the sequence.
Controls (otherwise VOID): C1 a synthetic rational ratio law is reproduced exactly at n = 8, 9; C2 a synthetic non-rational ratio law is not.
Usage: py q1752_pi2_distance_split.py (from small_gap_q206/; seconds)."""

import json, os, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
D = json.load(open(os.path.join(H, "..", "centre_scripts", "q1634_out_10.json")))
POLY = {}
for n in range(2, 11):
    ex = sp.expand(sp.sympify(D["e%d" % n].replace("pi", "PI"), locals={"PI": x}))
    POLY[n] = {m[0]: c for m, c in sp.Poly(ex, x).terms()}


def fit_check(q, fit_ns, check_ns, dn=2):
    n_ = sp.Symbol("n")
    k = dn + 1
    rows = [[q[m] * m**i for i in range(k)] + [-(m**i) for i in range(k)] for m in fit_ns]
    nsp = sp.Matrix(rows).nullspace()
    if len(nsp) != 1:
        return None, "nullspace dimension %d" % len(nsp)
    v = nsp[0]
    den = sum(v[i] * n_**i for i in range(k))
    num = sum(v[k + i] * n_**i for i in range(k))
    f = sp.simplify(num / den)
    ok = all(sp.simplify(f.subs(n_, m) - q[m]) == 0 for m in check_ns)
    return (ok, sp.factor(f)), "ok"


def run(c):
    q = {n: sp.Rational(c[n + 1]) / sp.Rational(c[n]) for n in range(3, 10)}
    res, msg = fit_check(q, range(3, 8), (8, 9))
    return q, res, msg


syn1 = {3: sp.Integer(1)}
syn2 = {3: sp.Integer(1)}
for n in range(3, 10):
    syn1[n + 1] = syn1[n] * sp.Rational((2 * n + 1) * (n + 3), (n + 2) * (3 * n - 1))
    syn2[n + 1] = syn2[n] * sp.Rational(n + 1, n + 2) * (1 + sp.Rational(1, 2**n))
_, f1, _ = run(syn1)
_, f2, _ = run(syn2)
c1 = f1 is not None and f1[0] is True
c2 = f2 is None or f2[0] is False
print(
    "C1 synthetic rational law reproduced exactly: %s; C2 synthetic non-rational law not reproduced: %s"
    % (c1, c2)
)
if not (c1 and c2):
    print("VOID: control failed")
    sys.exit(2)
out = {}
for m in (0, 1, 2):
    d = {n: POLY[n].get(2 * n - 2 - 2 * m, 0) for n in range(3, 11)}
    if any(d[n] == 0 for n in d):
        print("m = %d: a coefficient is zero, ratios undefined" % m)
        out[m] = (None, None)
        continue
    q, res, msg = run(d)
    out[m] = (res, msg)
    print(
        "m = %d: d_m(3..6) = %s; signs n = 3..10: %s"
        % (
            m,
            ", ".join(str(d[n]) for n in range(3, 7)),
            "".join("+" if d[n] > 0 else "-" for n in range(3, 11)),
        )
    )
    print("   ratios n = 3..9: " + "; ".join("%d: %.6f" % (n, float(q[n])) for n in range(3, 10)))
    print(
        "   fit on n = 3..7 -> %s"
        % (
            ("reproduces n = 8, 9: %s; function %s" % (res[0], res[1]))
            if res is not None
            else ("no unique fit (%s)" % msg)
        )
    )
lab = lambda q: "holds" if q else "does not hold"
ok = lambda m: out[m][0] is not None and out[m][0][0] is True
p1 = ok(1) or ok(2)
p2 = not ok(0)
print(
    "P1 at least one of m = 1, 2 reproduced exactly: %s (m = 1: %s, m = 2: %s); P2 m = 0 not reproduced: %s; P3: %s"
    % (
        lab(p1),
        lab(ok(1)),
        lab(ok(2)),
        lab(p2),
        (
            ("; ".join("m = %d: %s" % (m, out[m][0][1]) for m in (1, 2) if ok(m)))
            if p1
            else "not evaluated (P1 does not hold)"
        ),
    )
)
print("READING q1752 done (labels by the script; no verdict word)")
