"""q1775 (10 Oct 2026, REGISTRATION, no verdict word): prospective prediction of the lowest-power coefficient of the exact f32 = F_16 from the quadratic growth law of the diagonal coefficients d_k (q1774). The ratio a_k = d_(k+1)/d_k is described by a quadratic in k (error 0.26 percent for a_6 from k = 2..5). Rule fixed here, before f32 exists: (i) fit the quadratic alpha k^2 + beta k + gamma by least squares to a_k for k = 3..6 (prediction A) and for k = 2..6 (prediction B); (ii) a_7 = value of the quadratic at k = 7; d_8 = d_7 a_7; (iii) bhat(8) = coefficient of z^8 in exp(sum_(k<=8) d_k z^k), where bhat(K) = 8^(2K) b(2K) and b(j) is the coefficient of the lowest power u^ceil(j/2) = pi^-j of F_j (here j = 16, power pi^-16, u^8); (iv) the registered prediction is that the coefficient of pi^-16 in f32 times 8^16 lies between 0.97 times the smaller and 1.03 times the larger (in absolute value) of the two predictions A and B.
Usage: py q1775_prereg_f32_bottom.py (writes q1775_prereg_f32_bottom.txt with the values, the band and the sha256 of this script).
"""

import os, re, hashlib, datetime
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


def pred(ks):
    X = np.array([[k * k, k, 1.0] for k in ks])
    c, *_ = np.linalg.lstsq(X, np.array([a[k] for k in ks]), rcond=None)
    a7 = float(c[0] * 49 + c[1] * 7 + c[2])
    d8 = d[7] * sp.nsimplify(a7, rational=True)
    dd = dict(d)
    dd[8] = d8
    e = sp.series(sp.exp(sum(dd[k] * z**k for k in range(1, 9))), z, 0, 9).removeO()
    return a7, float(d8), float(e.coeff(z, 8))


pA = pred([3, 4, 5, 6])
pB = pred([2, 3, 4, 5, 6])
lo = min(abs(pA[2]), abs(pB[2])) * 0.97
hi = max(abs(pA[2]), abs(pB[2])) * 1.03
print("prediction A (k = 3..6): a_7 = %.4f, d_8 = %.6e, bhat(8) = %.6e" % pA)
print("prediction B (k = 2..6): a_7 = %.4f, d_8 = %.6e, bhat(8) = %.6e" % pB)
print(
    "registered band for |8^16 times the coefficient of pi^-16 in f32|: [%.6e, %.6e], sign negative"
    % (lo, hi)
)
me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
out = (
    "Registration q1775 at %s on %s by code (f32 does not exist yet; the run py q1557_f12_symbolic.py 32 started at about 14:00:30). Prediction: the coefficient of pi^-16 in f32 times 8^16 is negative and its absolute value lies in [%.6e, %.6e] (predictions A %.6e and B %.6e of bhat(8), from the quadratic growth law of the diagonal ratios, rule in the docstring of small_gap_q206/q1775_prereg_f32_bottom.py, sha256 %s). Evaluate with q1776 when q1557_stdout_ORD32.txt contains the line for f32.\n"
    % (
        datetime.datetime.now().strftime("%H:%M:%S"),
        datetime.date.today().isoformat(),
        lo,
        hi,
        pA[2],
        pB[2],
        me,
    )
)
open(os.path.join(H, "q1775_prereg_f32_bottom.txt"), "w", encoding="utf-8").write(out)
print(out.strip())
