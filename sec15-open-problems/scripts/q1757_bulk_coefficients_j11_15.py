"""q1757 (10 Oct 2026, READING, no verdict word): prospective test of the exact structure found in q1753 and q1756 on the bulk coefficients F_j for j = 11..15, which were NOT used to find it (F_11 = f22, F_12 = f24, F_13 = f26, F_14 = f28, F_15 = f30, exact forms in q1557_stdout_ORD30.txt, expanded in u = 1/pi^2).
The structure from j = 2..10: the coefficient of u^j is (1/8)^j; the coefficient of u^(j-1) times 8^j is Q(j) = -j (j - 1) (6 j^3 + j^2 - 19 j + 16) / 90; the lowest power is u^ceil(j/2).
Predictions (written before the run, with j = 11..15 never looked at):
 P1 the coefficient of u^(j-1) times 8^j equals Q(j) exactly for j = 11, 12, 13, 14, 15;
 P2 the coefficient of u^j equals (1/8)^j exactly for j = 11..15;
 P3 the lowest power in F_j is u^ceil(j/2) for j = 11..15.
Control (otherwise VOID): the same code applied to j = 2..10 (from births_law.py) gives P1 to P3 true there, and a changed Q (one coefficient altered) makes P1 false at j = 11.
Usage: py q1757_bulk_coefficients_j11_15.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
Q = lambda j, c=16: -sp.Rational(j * (j - 1) * (6 * j**3 + j**2 - 19 * j + c), 90)


def coeffs(expr):
    return {m[0]: sp.Rational(c) for m, c in sp.Poly(sp.expand(expr.subs(sp.pi, 1 / x)), x).terms()}


src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read().splitlines()
ns = {"PI": sp.pi}
for ln in src:
    if re.match(r"^(k1 =|c6 =|c10 =|f1[2-9] =|f20 =)", ln):
        exec(ln, ns)
F_old = {
    j: f
    for j, f in zip(
        range(2, 11),
        [ns["k4"], ns["c6"], ns["c8"], ns["c10"], ns["f12"], ns["f14"], ns["f16"], ns["f18"], ns["f20"]],
    )
}
txt = open(os.path.join(H, "q1557_stdout_ORD30.txt"), encoding="utf-8").read()
F_new = {}
for j, n in zip(range(11, 16), (22, 24, 26, 28, 30)):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    F_new[j] = sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi})


def check(F, js, c=16):
    p1 = all(coeffs(F[j]).get(2 * j - 2, 0) * 8**j == Q(j, c) for j in js)
    p2 = all(coeffs(F[j]).get(2 * j, 0) == sp.Rational(1, 8**j) for j in js)
    p3 = all(min(coeffs(F[j])) // 2 == -(-j // 2) for j in js)
    return p1, p2, p3


c_old = check(F_old, range(2, 11))
c_bad = check(F_new, [11], c=15)[0]
print(
    "CONTROL j = 2..10: P1 %s, P2 %s, P3 %s; altered Q makes P1 false at j = 11: %s" % (c_old + (not c_bad,))
)
if not (all(c_old) and not c_bad):
    print("VOID: control failed")
    sys.exit(2)
for j in range(11, 16):
    cf = coeffs(F_new[j])
    print(
        "j = %d: powers of 1/pi %s; coefficient of u^(j-1) times 8^j = %s; Q(j) = %s; coefficient of u^j times 8^j = %s"
        % (j, sorted(cf), cf.get(2 * j - 2, 0) * 8**j, Q(j), cf.get(2 * j, 0) * 8**j)
    )
p1, p2, p3 = check(F_new, range(11, 16))
lab = lambda q: "holds" if q else "does not hold"
print(
    "P1 coefficient of u^(j-1) times 8^j equals Q(j) for j = 11..15: %s; P2 top coefficient (1/8)^j: %s; P3 lowest power u^ceil(j/2): %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1757 done (labels by the script; no verdict word)")
