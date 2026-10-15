"""q1750 (10 Oct 2026, REGISTRATION, no verdict word): prospective prediction of the ratio of the constant coefficients c_0(11)/c_0(10) of the exact terms e_n = sum_j c_j(n) pi^(2j) (centre_scripts/q1634_out_10.json gives c_0(n) for n = 2..10). The exact e_11 (centre_scripts/q1634_out_11.json, run PID 4140) does not exist yet; its constant coefficient will be compared with the band below by q1751 once the file exists.
Rule (fixed here, before e_11 exists): the ratios s_n = c_0(n+1)/c_0(n) for n = 3..9 are extrapolated to n = 10 by three least-squares polynomials: linear on n = 7..9, quadratic on n = 6..9, quadratic on n = 5..9. The band is [min - 0.02, max + 0.02] of the three predictions. Registered prediction: the true s_10 = c_0(11)/c_0(10) lies inside the band.
Usage: py q1750_prereg_c0_ratio_11.py (from small_gap_q206/; seconds; prints the band and writes ../centre_scripts/q1750_prereg_c0_ratio.txt with the sha256 of this script).
"""

import json, os, hashlib, datetime
import numpy as np
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
D = json.load(open(os.path.join(H, "..", "centre_scripts", "q1634_out_10.json")))
c0 = {}
for n in range(3, 11):
    ex = sp.expand(sp.sympify(D["e%d" % n].replace("pi", "PI"), locals={"PI": x}))
    c0[n] = sp.Rational(sp.Poly(ex, x).coeff_monomial(1))
s = {n: float(c0[n + 1] / c0[n]) for n in range(3, 10)}
preds = []
for deg, lo in ((1, 7), (2, 6), (2, 5)):
    ns = np.arange(lo, 10)
    p = np.polyfit(ns, [s[n] for n in ns], deg)
    preds.append(float(np.polyval(p, 10)))
band = (min(preds) - 0.02, max(preds) + 0.02)
print("s_n = c_0(n+1)/c_0(n), n = 3..9: " + " ".join("%.6f" % s[n] for n in range(3, 10)))
print(
    "predictions of s_10 (linear 7..9, quadratic 6..9, quadratic 5..9): "
    + " ".join("%.5f" % q for q in preds)
)
print("registered band for s_10: [%.5f, %.5f]" % band)
me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
txt = (
    "Registration q1750 at %s on %s by code (the exact e_11 does not exist yet): the ratio s_10 = c_0(11)/c_0(10) of the constant coefficients of e_11 and e_10 lies in [%.5f, %.5f]. Rule in the docstring of small_gap_q206/q1750_prereg_c0_ratio_11.py (sha256 %s). Evaluate with q1751 when centre_scripts/q1634_out_11.json exists.\n"
    % (datetime.datetime.now().strftime("%H:%M:%S"), datetime.date.today().isoformat(), band[0], band[1], me)
)
open(os.path.join(H, "..", "centre_scripts", "q1750_prereg_c0_ratio.txt"), "w", encoding="utf-8").write(txt)
print(txt.strip())
