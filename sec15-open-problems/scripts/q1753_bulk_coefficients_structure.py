"""q1753 (10 Oct 2026, READING, no verdict word): exact structure of the bulk coefficients F_j (F_0 = 1, F_1 = k1, F_2 = k4, F_3 = c6, F_4 = c8, F_5 = c10, F_6 = f12, ..., F_10 = f20 of births_law.py) as polynomials in u = 1/pi^2 with rational coefficients. The definitions are taken from the source text of births_law.py (lines with k1, c6, c10, f12.., executed with pi as a sympy symbol), expanded in x = 1/pi.
NOT BLIND: the form of f12 (powers pi^-6 to pi^-12 with the top term 1/(262144 pi^12)) was seen when the task was written. Predictions (written at 13:44):
 P1 the coefficient of the highest power u^j of F_j equals exactly (1/8)^j for j = 2..10;
 P2 the coefficient of u^(j-1), times 8^j, is a polynomial in j of degree at most 2 determined exactly on j = 4..6 that reproduces j = 7..10 exactly;
 P3 the lowest power in F_j is u^ceil(j/2) for j = 2..10.
Controls (otherwise VOID): C1 a synthetic sequence with a known quadratic is recovered exactly by the same fit at j = 7..10; C2 a synthetic cubic is not.
Usage: py q1753_bulk_coefficients_structure.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
j_ = sp.Symbol("j")
src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read().splitlines()
ns = {"PI": sp.pi}
for ln in src:
    if re.match(r"^(k1 =|c6 =|c10 =|f1[2-9] =|f20 =)", ln):
        exec(ln, {}, ns) if False else exec(ln, ns)
F = [
    sp.Integer(1),
    ns["k1"],
    ns["k4"],
    ns["c6"],
    ns["c8"],
    ns["c10"],
    ns["f12"],
    ns["f14"],
    ns["f16"],
    ns["f18"],
    ns["f20"],
]
POLY = {}
for j in range(1, 11):
    e = sp.expand(sp.nsimplify(F[j]).subs(sp.pi, 1 / x)) if False else sp.expand(F[j].subs(sp.pi, 1 / x))
    P = sp.Poly(e, x)
    POLY[j] = {m[0]: sp.Rational(c) for m, c in P.terms()}
print("powers of 1/pi present: " + "; ".join("F_%d: %s" % (j, sorted(POLY[j])) for j in range(1, 11)))


def quad_fit(seq, fit_js, check_js):
    pts = [(j, seq[j]) for j in fit_js]
    poly = sp.expand(sp.interpolate(pts, j_))
    return poly, all(poly.subs(j_, j) == seq[j] for j in check_js), sp.degree(poly, j_) if poly != 0 else 0


syn1 = {j: sp.Rational(3 * j * j - 5 * j + 7, 11) for j in range(4, 11)}
syn2 = {j: sp.Rational(j**3 + 2 * j, 5) for j in range(4, 11)}
p1c, ok1, _ = quad_fit(syn1, (4, 5, 6), (7, 8, 9, 10))
p2c, ok2, _ = quad_fit(syn2, (4, 5, 6), (7, 8, 9, 10))
print("C1 synthetic quadratic reproduced: %s; C2 synthetic cubic not reproduced: %s" % (ok1, not ok2))
if not (ok1 and not ok2):
    print("VOID: control failed")
    sys.exit(2)
top = {j: POLY[j].get(2 * j, 0) for j in range(2, 11)}
p1 = all(top[j] == sp.Rational(1, 8**j) for j in range(2, 11))
print(
    "top coefficient (power u^j) times 8^j for j = 2..10: "
    + " ".join(str(top[j] * 8**j) for j in range(2, 11))
)
sec = {j: POLY[j].get(2 * j - 2, 0) * 8**j for j in range(2, 11)}
print(
    "second coefficient (power u^(j-1)) times 8^j for j = 2..10: "
    + "; ".join("%d: %s" % (j, sec[j]) for j in range(2, 11))
)
poly, ok, deg = quad_fit(sec, (4, 5, 6), (7, 8, 9, 10))
print("quadratic through j = 4..6: %s; reproduces j = 7..10: %s" % (poly, ok))
low = {j: min(POLY[j]) // 2 for j in range(2, 11)}
p3 = all(low[j] == -(-j // 2) for j in range(2, 11))
print(
    "lowest power u^k for j = 2..10: "
    + " ".join("%d:%d" % (j, low[j]) for j in range(2, 11))
    + "; ceil(j/2): "
    + " ".join("%d:%d" % (j, -(-j // 2)) for j in range(2, 11))
)
lab = lambda q: "holds" if q else "does not hold"
print(
    "P1 top coefficient equals (1/8)^j for j = 2..10: %s; P2 second coefficient times 8^j reproduced by a quadratic in j: %s; P3 lowest power u^ceil(j/2): %s"
    % (lab(p1), lab(ok), lab(p3))
)
print("READING q1753 done (labels by the script; no verdict word)")
