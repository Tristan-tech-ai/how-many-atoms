"""q1774 (10 Oct 2026, READING, no verdict word): growth law of the diagonal coefficients d_k of the log levels (q1772). If d_k grows like a factorial of 2k with the bulk rate A, the ratio a_k = d_(k+1)/d_k tends to alpha k^2 with alpha = 256 pi^2 / A^2 (the factor 64 from the scaling 8^(2K) of the bottom coefficients, pi^2 from u^-1, 4 k^2 from the two linear factors per step).
NOT BLIND: the ratios 2.33333, 14.73016, 29.16509, 47.49739, 69.30329, 94.65655 for k = 1..6 were printed in q1772. Predictions (written at 14:16):
 P1 a least-squares quadratic alpha k^2 + beta k + gamma through the ratios k = 3..6 has alpha between 1.5 and 2.6;
 P2 alpha lies within 25 percent of 256 pi^2 / 32.307^2 = 2.42 (the Stokes-rate value is 2.33);
 P3 the quadratic fitted on k = 2..5 predicts the ratio at k = 6 within 5 percent.
Controls (otherwise VOID): C1 a synthetic ratio sequence 2 k^2 + 3 k + 1 is recovered exactly by the same fit; C2 the same sequence with an added 1/k term is not recovered to 1e-6.
Usage: py q1774_dk_growth.py (from small_gap_q206/; seconds)."""

import os, re, sys
import numpy as np
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
D = sp.series(sp.log(sum(bhat[K] * z**K for K in range(0, 8))), z, 0, 8).removeO()
d = {k: sp.Rational(D.coeff(z, k)) for k in range(1, 8)}
a = {k: float(d[k + 1] / d[k]) for k in range(1, 7)}


def quad(ks, vals):
    X = np.array([[k * k, k, 1.0] for k in ks])
    c, *_ = np.linalg.lstsq(X, np.array(vals), rcond=None)
    return c


ks = [1, 2, 3, 4, 5]
c1 = quad(ks, [2 * k * k + 3 * k + 1 for k in ks])
c1ok = np.allclose(c1, [2, 3, 1], atol=1e-9)
c2 = quad(ks, [2 * k * k + 3 * k + 1 + 1.0 / k for k in ks])
c2ok = not np.allclose(c2, [2, 3, 1], atol=1e-6)
print("C1 synthetic quadratic recovered: %s; C2 synthetic with a 1/k term not recovered: %s" % (c1ok, c2ok))
if not (c1ok and c2ok):
    print("VOID: control failed")
    sys.exit(2)
print("ratios a_k = d_(k+1)/d_k, k = 1..6: " + " ".join("%.5f" % a[k] for k in range(1, 7)))
cA = quad([3, 4, 5, 6], [a[k] for k in (3, 4, 5, 6)])
cB = quad([2, 3, 4, 5], [a[k] for k in (2, 3, 4, 5)])
pred6 = float(np.dot([36, 6, 1], cB))
print(
    "quadratic on k = 3..6: alpha %.4f, beta %.4f, gamma %.4f; on k = 2..5: alpha %.4f, beta %.4f, gamma %.4f; prediction of a_6 from k = 2..5: %.5f against %.5f (error %.2f percent)"
    % (*cA, *cB, pred6, a[6], 100 * abs(pred6 / a[6] - 1))
)
al_ref = 256 * np.pi**2 / 32.307**2
al_st = 256 * np.pi**2 / 32.957**2
print("reference alpha = 256 pi^2 / A^2: %.4f for A = 32.307, %.4f for A = 32.957" % (al_ref, al_st))
lab = lambda q: "holds" if q else "does not hold"
p1 = 1.5 <= cA[0] <= 2.6
p2 = abs(cA[0] / al_ref - 1) <= 0.25
p3 = abs(pred6 / a[6] - 1) <= 0.05
print(
    "P1 alpha of the k = 3..6 quadratic in [1.5, 2.6]: %s (alpha = %.4f); P2 alpha within 25 percent of %.4f: %s; P3 prediction of a_6 within 5 percent: %s"
    % (lab(p1), cA[0], al_ref, lab(p2), lab(p3))
)
print("READING q1774 done (labels by the script; no verdict word)")
