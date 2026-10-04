"""q855 (03 Oct 2026, copy of scratchpad fit_k.py, Amendment 1520): fit the strip response (k, k2) of Amendment 1519 to the accepted q850 rows at A = 16 and 20.3 (reads logs; writes nothing)."""

import re, math
import numpy as np
from scipy.optimize import least_squares

CS = r"centre_scripts"
beta = 3.98160794709
h1 = 0.5 * math.log(2 * math.pi * math.e)
pat = re.compile(
    r"A ([0-9.]+) t ([0-9.]+): .*residual ([0-9.e+-]+); KKT max ([-+0-9.e]+); I \+ tE = ([0-9.]+)"
)
rows = {}
for f in ("q850_seed19_fine.log", "q850_seed19.log", "q850_A20.3_seed25.log"):
    for line in open(CS + "\\" + f):
        m = pat.search(line)
        if m and float(m.group(3)) < 1e-10 and float(m.group(4)) < 1e-9:
            rows[(float(m.group(1)), float(m.group(2)))] = float(m.group(5))
print("accepted rows:", sorted(rows))
pts = [(A, t, C) for (A, t), C in rows.items() if t > 0]


def model(p, A, t, order):
    k, k2 = p[0], (p[1] if order >= 2 else 0.0)
    xr = A + beta / 2 + k * t + k2 * t * t
    xl = A + beta / 2 - k * t + k2 * t * t
    return math.log((math.exp(t * xr) - math.exp(-t * xl)) / t)


for order, tmax in ((2, 0.03), (2, 0.05), (1, 0.02)):
    sel = [(A, t, C) for A, t, C in pts if t <= tmax]
    res = lambda p: np.array([model(p, A, t, order) - (C + h1) for A, t, C in sel])
    sol = least_squares(res, x0=[-0.6, 0.0][:order] if order == 2 else [-0.6], xtol=1e-14, ftol=1e-14)
    print(
        f"order {order}, t <= {tmax}: n {len(sel)}; k = {sol.x[0]:+.5f}"
        + (f", k2 = {sol.x[1]:+.3f}" if order == 2 else "")
        + f"; max residual {np.abs(res(sol.x)).max():.1e}"
    )
for A in (16.0, 20.3):
    if (A, 0.0) in rows:
        print(
            f"t = 0 check A {A}: e^(C+h) - (2A + beta) = {math.exp(rows[(A, 0.0)] + h1) - (2*A + beta):+.2e}"
        )
print("predicted k = -0.6333; a = 2.48165 + k")
