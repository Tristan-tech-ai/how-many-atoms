"""q1772 (10 Oct 2026, READING, no verdict word): the diagonal coefficients d_k of the log levels. In ln F = -ln(1-x) + sum_k u^-k L_k(y) with y = x/(1-x) the level L_k starts at y^(2k). The coefficient of x^(2K) u^-K in F is the bottom coefficient c_K(2K) = 8^(2K) b(2K) of F_(2K) (b(j) = coefficient of the lowest power u^ceil(j/2), reading 206). Only the diagonal terms d_k z^k with z = x^2/u contribute to that order, so sum_K 8^(2K) b(2K) z^K = exp(sum_k d_k z^k), and d_k are the coefficients of ln of the bottom generating function. From q1771: d_1 = -2/3 and d_2 = -14/9 (the lowest coefficients of L_1 and L_2 in y).
NOT BLIND: d_1 and d_2 and the values b(j) are known; d_3 to d_7 were not computed. Predictions (written before the run):
 P1 the code reproduces d_1 = -2/3 and d_2 = -14/9 exactly (this is the control, and it is also the registered check of the identity);
 P2 d_k is negative for k = 3..7;
 P3 the numerators of d_k for k = 1..7 contain no prime above 31 (the numerators of b(j) do, reading 211).
Control (otherwise VOID): P1.
Usage: py q1772_diagonal_log.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
z = sp.Symbol("z")


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
bhat = {0: sp.Integer(1)}
for K in range(1, 8):
    cf = coeffs(F[2 * K])
    bhat[K] = cf[min(cf)] * 8 ** (2 * K)
B = sum(bhat[K] * z**K for K in range(0, 8))
D = sp.series(sp.log(B), z, 0, 8).removeO()
d = {k: sp.Rational(D.coeff(z, k)) for k in range(1, 8)}
lab = lambda q: "holds" if q else "does not hold"
c1 = d[1] == sp.Rational(-2, 3) and d[2] == sp.Rational(-14, 9)
print("CONTROL d_1 = -2/3 and d_2 = -14/9: %s" % lab(c1))
if not c1:
    print("VOID: control failed")
    sys.exit(2)
print("8^(2K) b(2K), K = 1..7: " + ", ".join(str(bhat[K]) for K in range(1, 8)))
print("d_k, k = 1..7: " + "; ".join("%d: %s (%.6g)" % (k, d[k], float(d[k])) for k in range(1, 8)))
print("ratios d_(k+1)/d_k, k = 1..6: " + " ".join("%.5f" % float(d[k + 1] / d[k]) for k in range(1, 7)))
big = {k: sorted(p for p in sp.factorint(abs(d[k].p)) if p > 31) for k in range(1, 8)}
bigden = {k: sorted(p for p in sp.factorint(d[k].q) if p > 31) for k in range(1, 8)}
print(
    "primes above 31 in the numerators: %s; in the denominators: %s"
    % ({k: v for k, v in big.items() if v} or "none", {k: v for k, v in bigden.items() if v} or "none")
)
print(
    "factored d_k: "
    + "; ".join("%d: %s / %s" % (k, sp.factorint(abs(d[k].p)), sp.factorint(d[k].q)) for k in range(1, 8))
)
p2 = all(d[k] < 0 for k in range(3, 8))
p3 = all(not v for v in big.values())
print(
    "P1 control d_1, d_2: %s; P2 d_k negative for k = 3..7: %s; P3 no prime above 31 in the numerators of d_k: %s"
    % (lab(c1), lab(p2), lab(p3))
)
print("READING q1772 done (labels by the script; no verdict word)")
