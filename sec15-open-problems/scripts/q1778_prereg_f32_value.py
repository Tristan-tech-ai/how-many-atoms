"""q1778 (10 Oct 2026, REGISTRATION, no verdict word): prospective prediction of the numeric value of f32 = F_16 from the bottom coefficient (q1775) and the share of the bottom power (q1777). Rule fixed here, before f32 exists:
 (i) the bottom term of F_16 is T16 = bhat(8) / 8^16 * pi^-16, with bhat(8) the mean of the predictions A and B of q1775 (-2.557337e+10 and -2.561941e+10);
 (ii) the share s_j = (bottom term of F_j) / F_j for the even j = 10, 12, 14 (q1777) is extrapolated to j = 16 by the least-squares line through (10, s_10), (12, s_12), (14, s_14);
 (iii) the predicted value is F_16 = T16 / s_16; the band is the prediction times 0.92 to 1.08 (negative);
 (iv) for comparison only, the ratio model of q1732 (F_j = F_(j-1) (j - 0.023) / 32.307) gives F_16 = F_15 x 0.4945; the registered statement is about the band.
Usage: py q1778_prereg_f32_value.py (writes q1778_prereg_f32_value.txt with the numbers and the sha256 of this script).
"""

import os, re, hashlib, datetime
import numpy as np
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")


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
u = 1 / mp.pi**2
sh = {}
for j in (10, 12, 14):
    cf = coeffs(F[j])
    pw = sorted(cf)
    b0 = mp.mpf(sp.N(cf[pw[0]], 50)) * u ** (pw[0] // 2)
    sh[j] = float(b0 / mp.mpf(sp.N(F[j], 50)))
c = np.polyfit([10, 12, 14], [sh[j] for j in (10, 12, 14)], 1)
s16 = float(np.polyval(c, 16))
bh = (-2.557337e10 + -2.561941e10) / 2
T16 = float(bh / mp.mpf(8) ** 16 * u**8)
pred = T16 / s16
F15 = float(mp.mpf(sp.N(F[15], 50)))
ratio_model = F15 * (16 - 0.023) / 32.307
lo, hi = pred * 1.08, pred * 0.92
print(
    "shares j = 10, 12, 14: %s; line extrapolated to j = 16: s_16 = %.5f"
    % (", ".join("%.5f" % sh[j] for j in (10, 12, 14)), s16)
)
print(
    "bottom term T16 = %.6e; predicted F_16 = T16 / s_16 = %.6e; band [%.6e, %.6e]; ratio-model value for comparison %.6e"
    % (T16, pred, lo, hi, ratio_model)
)
me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
out = (
    "Registration q1778 at %s on %s by code (f32 does not exist yet; the run py q1557_f12_symbolic.py 32 started at about 14:00:30). Prediction: the numeric value of f32 = F_16 lies in [%.6e, %.6e] (negative), from the bottom term %.6e of q1775 divided by the extrapolated share %.5f of the bottom power; the ratio-model value %.6e is given for comparison only. Rule in the docstring of small_gap_q206/q1778_prereg_f32_value.py (sha256 %s). Evaluate with q1779 when q1557_stdout_ORD32.txt contains the line for f32.\n"
    % (
        datetime.datetime.now().strftime("%H:%M:%S"),
        datetime.date.today().isoformat(),
        lo,
        hi,
        T16,
        s16,
        ratio_model,
        me,
    )
)
open(os.path.join(H, "q1778_prereg_f32_value.txt"), "w", encoding="utf-8").write(out)
print(out.strip())
