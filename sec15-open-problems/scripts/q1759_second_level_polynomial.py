"""q1759 (10 Oct 2026, READING, no verdict word): the coefficient c_2(j) of u^(j-2) in the bulk coefficient F_j times 8^j, for j = 4..15 (F_2..F_10 from births_law.py, F_11..F_15 from q1557_stdout_ORD30.txt). q1758 found no polynomial of degree <= 9 without further assumption. The coefficient of u^(j-2) is absent from F_j for j = 0, 1, 2, 3 (the power u^(j-2) is negative or below the lowest power u^ceil(j/2)), exactly as the factor j (j - 1) of the k = 1 polynomial Q(j) reflects the absent powers at j = 0, 1. Assumption A: c_2(j) 8^j = j (j-1) (j-2) (j-3) R(j) with R a polynomial in j.
NOT BLIND: q1756 to q1758 were read; the k = 2 sequence itself was printed in q1758's docstring-free data only as 'no polynomial of degree <= 9'.
Predictions (written before the run):
 P1 under assumption A the smallest degree of R for which the interpolation through the first deg+1 points of j = 4..15 reproduces all others exactly is 6 (total degree 10), with 5 independent checks;
 P2 the leading coefficient of c_2(j) 8^j equals half the square of the leading coefficient of Q(j) (the k = 1 polynomial has leading coefficient -1/15, so the square over two is 1/450).
Control (otherwise VOID): the same code applied to a synthetic sequence j (j-1)(j-2)(j-3) R(j) with a known quartic R is found at degree 4, and a sequence with a 2^j term is not found.
Usage: py q1759_second_level_polynomial.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
j_ = sp.Symbol("j")


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
CF = {j: coeffs(F[j]) for j in F}
W = lambda j: j * (j - 1) * (j - 2) * (j - 3)


def min_degree_R(seq, js, need=2):
    r = {j: seq[j] / W(j) for j in js}
    for d in range(0, len(js) - need):
        pts = [(j, r[j]) for j in js[: d + 1]]
        poly = sp.expand(sp.interpolate(pts, j_)) if d > 0 else sp.Integer(r[js[0]])
        if all(poly.subs(j_, j) == r[j] for j in js[d + 1 :]):
            return d, poly, len(js) - d - 1
    return None, None, 0


js = list(range(4, 16))
synR = lambda j: sp.Rational(j**4 - 2 * j**2 + 3, 5)
sa = {j: W(j) * synR(j) for j in js}
sb = {j: W(j) * (synR(j) + sp.Rational(2) ** j) for j in js}
da = min_degree_R(sa, js)[0]
db = min_degree_R(sb, js)[0]
print("CONTROL synthetic quartic R found at degree %s; synthetic with 2^j found: %s" % (da, db))
if da != 4 or db is not None:
    print("VOID: control failed")
    sys.exit(2)
seq = {j: CF[j].get(2 * (j - 2), 0) * 8**j for j in js}
print("c_2(j) * 8^j, j = 4..15: " + ", ".join(str(seq[j]) for j in js))
d, R, nchk = min_degree_R(seq, js)
lab = lambda q: "holds" if q else "does not hold"
if d is None:
    print("no polynomial R of degree <= %d reproduces the sequence with two checks" % (len(js) - 3))
    p1 = p2 = False
else:
    full = sp.expand(W(j_) * R)
    lead = sp.Poly(full, j_).LC()
    print("R of degree %d with %d independent checks; c_2(j) 8^j = %s" % (d, nchk, sp.factor(full)))
    print(
        "leading coefficient of c_2(j) 8^j: %s; square of the k = 1 leading coefficient over two: %s"
        % (lead, sp.Rational(1, 15) ** 2 / 2)
    )
    p1 = d == 6 and nchk == 5
    p2 = lead == sp.Rational(1, 15) ** 2 / 2
print("P1 degree of R is 6 with 5 checks: %s; P2 leading coefficient equals 1/450: %s" % (lab(p1), lab(p2)))
print("READING q1759 done (labels by the script; no verdict word)")
