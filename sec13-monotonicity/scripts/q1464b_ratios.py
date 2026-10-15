"""q1464b (9 Oct 2026, READING, no verdict): ratios a2/a1 and a3/a1 of the w-series (powers 1, 2, 3 of w = 1/v_c^2) of the six edge fractions, rows K 83..114. Usage: py q1464b_ratios.py (from small_gap_q206/)."""

import json, numpy as np
from scipy.optimize import brentq

pi = np.pi
rho_c = lambda K: brentq(
    lambda r: 4 * pi**2 * r**4 - r**2 + np.log(r) / 6 + 0.5482 - (K + 1 / 6) / 2, 0.05, 20
)
d = json.load(open("q1463_terms.json"))
ks = sorted(int(k) for k in d)
print(
    "READING q1464b: ratios a2/a1 and a3/a1 of the w-series (powers 1,2,3 of w = 1/v_c^2) of the six edge fractions, rows K 83..114"
)
for par in ("even", "odd"):
    sel = [k for k in ks if d[str(k)]["parity"] == par]
    w = np.array([1 / (4 * pi * rho_c(k)) ** 2 for k in sel])
    X = np.array([d[str(k)]["x_six"] for k in sel])
    BD = np.array([d[str(k)]["bd_six"] for k in sel])
    T = X * BD
    F = T / T[:, 3:].sum(axis=1)[:, None]
    M = np.vstack([np.ones_like(w), w, w**2, w**3]).T
    for j in range(6):
        c = np.linalg.lstsq(M, F[:, j], rcond=None)[0]
        print(f"  {par} fraction {j}: a1={c[1]:+.4f} a2/a1={c[2] / c[1]:+.2f} a3/a1={c[3] / c[1]:+.0f}")
