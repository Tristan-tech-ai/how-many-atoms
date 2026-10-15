"""q1756 (10 Oct 2026, FACT TABLE after q1753, post hoc, no predictions, no verdict word): the coefficient c_k(j) of u^(j-k) in the bulk coefficient F_j (u = 1/pi^2), multiplied by 8^j, for k = 1, 2, 3 and the valid range of j (the power u^(j-k) must not lie below the lowest power u^ceil(j/2), so j >= 2k). The finite differences of the sequence in j are printed, and for each k the smallest degree d is searched for which the exact interpolation polynomial through the first d+1 points reproduces all remaining points exactly; a degree is accepted only with at least two independent checks.
Post hoc: the integers 3 * c_1(j) * 8^j (j = 2..10) were seen in q1753 to have constant fifth differences; this script tests that exactly, and tries k = 2, 3.
Control: a synthetic sequence of degree 4 is found at degree 4 with the same code, and a sequence with a non-polynomial term (2^j) is not found.
Usage: py q1756_bulk_coefficient_polynomials.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
j_ = sp.Symbol("j")
src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read().splitlines()
ns = {"PI": sp.pi}
for ln in src:
    if re.match(r"^(k1 =|c6 =|c10 =|f1[2-9] =|f20 =)", ln):
        exec(ln, ns)
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
POLY = {
    j: {m[0]: sp.Rational(c) for m, c in sp.Poly(sp.expand(F[j].subs(sp.pi, 1 / x)), x).terms()}
    for j in range(1, 11)
}


def min_degree(seq, js, need=2):
    for d in range(0, len(js) - need):
        pts = [(j, seq[j]) for j in js[: d + 1]]
        poly = sp.expand(sp.interpolate(pts, j_)) if d > 0 else sp.Integer(seq[js[0]])
        if all(poly.subs(j_, j) == seq[j] for j in js[d + 1 :]):
            return d, poly, len(js) - d - 1
    return None, None, 0


syn_a = {j: sp.Rational(j**4 - 3 * j**2 + 5, 7) for j in range(2, 11)}
syn_b = {j: sp.Rational(j**2 + 2**j, 3) for j in range(2, 11)}
da, _, _ = min_degree(syn_a, list(range(2, 11)))
db, _, _ = min_degree(syn_b, list(range(2, 11)))
print("CONTROL synthetic degree 4 found at degree %s; synthetic with 2^j found: %s" % (da, db))
if da != 4 or db is not None:
    print("VOID: control failed")
    sys.exit(2)
for k in (1, 2, 3):
    js = list(range(2 * k, 11))
    seq = {j: POLY[j].get(2 * (j - k), 0) * 8**j for j in js}
    print("k = %d: c_k(j) * 8^j for j = %d..10: %s" % (k, js[0], ", ".join(str(seq[j]) for j in js)))
    vals = [seq[j] for j in js]
    diffs = [vals]
    while len(diffs[-1]) > 1:
        diffs.append([b - a for a, b in zip(diffs[-1][:-1], diffs[-1][1:])])
    print(
        "   finite differences, last entry of each order: "
        + " ".join(str(dd[-1]) for dd in diffs[1:])
        + "; constant rows: "
        + str([i for i, dd in enumerate(diffs) if len(dd) >= 3 and len(set(dd)) == 1])
    )
    d, poly, nchk = min_degree(seq, js)
    if d is None:
        print("   no polynomial in j of degree <= %d reproduces the sequence with two checks" % (len(js) - 3))
    else:
        print(
            "   exact polynomial of degree %d with %d independent checks: %s ; factored: %s"
            % (d, nchk, poly, sp.factor(poly))
        )
print("TABLE q1756 done (facts; no labels)")
