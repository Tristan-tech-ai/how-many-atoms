"""q1758 (10 Oct 2026, FACT TABLE after q1757, post hoc, no predictions, no verdict word): the coefficients c_k(j) of u^(j-k) in the bulk coefficient F_j times 8^j, for k = 1..4 and j up to 15 (F_2..F_10 from births_law.py, F_11..F_15 from q1557_stdout_ORD30.txt). For each k the smallest degree d is searched for which the exact interpolation polynomial through the first d+1 points reproduces every remaining point exactly, with at least two independent checks.
Control: q1756's k = 1 polynomial is found again at degree 5, and a synthetic degree-4 sequence is found at degree 4.
Usage: py q1758_bulk_coefficient_polynomials_ext.py (from small_gap_q206/; seconds)."""

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


def min_degree(seq, js, need=2):
    for d in range(0, len(js) - need):
        pts = [(j, seq[j]) for j in js[: d + 1]]
        poly = sp.expand(sp.interpolate(pts, j_)) if d > 0 else sp.Integer(seq[js[0]])
        if all(poly.subs(j_, j) == seq[j] for j in js[d + 1 :]):
            return d, poly, len(js) - d - 1
    return None, None, 0


Q1 = lambda j: -j * (j - 1) * (6 * j**3 + j**2 - 19 * j + 16) / sp.Integer(90)
s1 = {j: CF[j].get(2 * (j - 1), 0) * 8**j for j in range(2, 16)}
d1, p1, n1 = min_degree(s1, list(range(2, 16)))
syn = {j: sp.Rational(j**4 - 3 * j**2 + 5, 7) for j in range(2, 16)}
ds, _, _ = min_degree(syn, list(range(2, 16)))
ctl = d1 == 5 and sp.simplify(p1 - Q1(j_)) == 0 and ds == 4
print(
    "CONTROL k = 1 polynomial found again at degree %s with %d checks, equal to Q: %s; synthetic degree 4 found at %s"
    % (d1, n1, sp.simplify(p1 - Q1(j_)) == 0, ds)
)
if not ctl:
    print("VOID: control failed")
    sys.exit(2)
for k in (2, 3, 4):
    js = list(range(2 * k, 16))
    seq = {j: CF[j].get(2 * (j - k), 0) * 8**j for j in js}
    d, poly, nchk = min_degree(seq, js)
    print(
        "k = %d (j = %d..15, %d points): %s"
        % (
            k,
            js[0],
            len(js),
            (
                ("exact polynomial of degree %d with %d independent checks: %s" % (d, nchk, sp.factor(poly)))
                if d is not None
                else "no polynomial of degree <= %d reproduces the sequence with two checks" % (len(js) - 3)
            ),
        )
    )
print("TABLE q1758 done (facts; no labels)")
