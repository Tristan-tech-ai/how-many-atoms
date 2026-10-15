"""q1754 (10 Oct 2026, REGISTRATION, no verdict word): prospective prediction of the ratio d_1(11)/d_1(10) of the coefficients of pi^(2n-4) (distance m = 1 from the top power) of the exact terms e_11 and e_10. The exact e_11 (centre_scripts/q1634_out_11.json, run PID 4140) does not exist yet; q1755 compares it with the band below.
Rule (fixed before e_11 exists): s_n = d_1(n+1)/d_1(n) for n = 3..9 from centre_scripts/q1634_out_10.json; the prediction for s_10 is the linear extrapolation of s_8 and s_9; the band is the prediction plus or minus 0.12 (about 2.6 times the last increment). Registered prediction: the true s_10 lies inside the band.
Usage: py q1754_prereg_d1_ratio_11.py (from small_gap_q206/; writes ../centre_scripts/q1754_prereg_d1_ratio.txt with the sha256 of this script).
"""

import json, os, hashlib, datetime
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
D = json.load(open(os.path.join(H, "..", "centre_scripts", "q1634_out_10.json")))
d1 = {}
for n in range(3, 11):
    ex = sp.expand(sp.sympify(D["e%d" % n].replace("pi", "PI"), locals={"PI": x}))
    d1[n] = sp.Rational(sp.Poly(ex, x).coeff_monomial(x ** (2 * n - 4)))
s = {n: float(d1[n + 1] / d1[n]) for n in range(3, 10)}
pred = s[9] + (s[9] - s[8])
band = (pred - 0.12, pred + 0.12)
print("s_n = d_1(n+1)/d_1(n), n = 3..9: " + " ".join("%.6f" % s[n] for n in range(3, 10)))
print(
    "prediction of s_10 (linear extrapolation of s_8, s_9): %.5f; band [%.5f, %.5f]"
    % (pred, band[0], band[1])
)
me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
txt = (
    "Registration q1754 at %s on %s by code (the exact e_11 does not exist yet): the ratio s_10 = d_1(11)/d_1(10) of the coefficients of pi^18 in e_11 and pi^16 in e_10 lies in [%.5f, %.5f]. Rule in the docstring of small_gap_q206/q1754_prereg_d1_ratio_11.py (sha256 %s). Evaluate with q1755 when centre_scripts/q1634_out_11.json exists.\n"
    % (datetime.datetime.now().strftime("%H:%M:%S"), datetime.date.today().isoformat(), band[0], band[1], me)
)
open(os.path.join(H, "..", "centre_scripts", "q1754_prereg_d1_ratio.txt"), "w", encoding="utf-8").write(txt)
print(txt.strip())
