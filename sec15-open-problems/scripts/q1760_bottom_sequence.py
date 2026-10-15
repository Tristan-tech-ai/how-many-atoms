"""q1760 (10 Oct 2026, READING, no verdict word): the bottom sequence b(j) = coefficient of the lowest power u^ceil(j/2) (u = 1/pi^2) of the bulk coefficient F_j, j = 2..15 (F_2..F_10 from births_law.py, F_11..F_15 from q1557_stdout_ORD30.txt). The factorial growth of the bulk coefficients comes from the lowest powers, not from the top levels of q1753 to q1759.
NOT BLIND: the lowest powers are known from q1753; the values b(j) were not printed before. Predictions (written at 13:51):
 P1 for each parity of j separately the ratios b(j+2)/b(j) for j = 2..13 are reproduced exactly by a rational function of degree (1, 1) determined on the first three ratios, with the other three as checks;
 P2 the sign of b(j) is periodic in j;
 P3 |b(j+2)/b(j)| grows with j.
Controls (otherwise VOID): C1 a synthetic sequence with a known (1, 1) ratio law is reproduced exactly; C2 a synthetic sequence with a (2, 2) law is not reproduced by the (1, 1) fit.
Usage: py q1760_bottom_sequence.py (from small_gap_q206/; seconds)."""

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


def fit_check(q, fit_js, check_js, dn=1):
    k = dn + 1
    rows = [[q[m] * m**i for i in range(k)] + [-(m**i) for i in range(k)] for m in fit_js]
    nsp = sp.Matrix(rows).nullspace()
    if len(nsp) != 1:
        return None
    v = nsp[0]
    den = sum(v[i] * n_**i for i in range(k))
    num = sum(v[k + i] * n_**i for i in range(k))
    f = sp.simplify(num / den)
    return all(sp.simplify(f.subs(n_, m) - q[m]) == 0 for m in check_js), sp.factor(f)


def test(seq, par):
    js = [j for j in range(2, 14) if j % 2 == par]
    q = {j: seq[j + 2] / seq[j] for j in js}
    return fit_check(q, js[:3], js[3:]), q


s1 = {}
s2 = {}
for j in range(2, 16):
    s1[j] = sp.Integer(1) if j < 4 else None
syn1 = {2: sp.Integer(1), 3: sp.Integer(2)}
syn2 = dict(syn1)
for j in range(2, 14):
    syn1[j + 2] = syn1[j] * sp.Rational(3 * j + 1, 2 * j + 5)
    syn2[j + 2] = syn2[j] * sp.Rational((j + 1) * (j + 4), (j + 2) * (2 * j + 3))
c1 = all(test(syn1, p)[0] is not None and test(syn1, p)[0][0] for p in (0, 1))
c2 = all(test(syn2, p)[0] is None or not test(syn2, p)[0][0] for p in (0, 1))
print(
    "C1 synthetic (1, 1) law reproduced: %s; C2 synthetic (2, 2) law not reproduced by the (1, 1) fit: %s"
    % (c1, c2)
)
if not (c1 and c2):
    print("VOID: control failed")
    sys.exit(2)
print("b(j), j = 2..15: " + "; ".join("%d: %s" % (j, b[j]) for j in range(2, 16)))
print("signs j = 2..15: " + "".join("+" if b[j] > 0 else "-" for j in range(2, 16)))
res = {}
for par in (0, 1):
    r, q = test(b, par)
    res[par] = r
    print(
        "parity %s: ratios b(j+2)/b(j): %s"
        % ("even" if par == 0 else "odd", "; ".join("%d: %.5f" % (j, float(q[j])) for j in sorted(q)))
    )
    print(
        "   (1,1) fit on the first three, checks on the others -> %s"
        % (("reproduces: %s; function %s" % r) if r is not None else "no unique fit")
    )
lab = lambda q: "holds" if q else "does not hold"
sg = [1 if b[j] > 0 else -1 for j in range(2, 16)]
per = [p for p in range(1, 8) if all(sg[i] == sg[i + p] for i in range(len(sg) - p))]
absr = [float(abs(b[j + 2] / b[j])) for j in range(2, 14)]
grow = all(absr[i + 2] > absr[i] for i in range(len(absr) - 2))
p1 = all(res[p] is not None and res[p][0] for p in (0, 1))
print(
    "P1 degree (1, 1) ratio law for both parities: %s; P2 sign periodic with the period(s) %s: %s; P3 |ratio| grows with j (compared two steps apart): %s"
    % (lab(p1), per, lab(bool(per)), lab(grow))
)
print("READING q1760 done (labels by the script; no verdict word)")
