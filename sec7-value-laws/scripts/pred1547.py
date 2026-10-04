"""1547 predictions: NPMLE eccentric annulus R 18, r 8 at e = 3 (value and ring-count changes), from the merged q858 scans."""

import json, math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

SP = r"scratch"
d = {}
for f in ("delta0_scan_q858.json", "delta0_scan_q858_ext.json"):
    for a, v in json.load(open(f"{SP}\\{f}")).items():
        if v["acc"]:
            d[float(a)] = v["delta0"]
A = np.array(sorted(d))
Yv = np.array([d[a] for a in A])
sp = CubicSpline(A, Yv)
print("scan points", len(A), A.min(), A.max())
R, r = 18.0, 8.0
th = np.linspace(0, 2 * np.pi, 40001)[:-1]
dth = 2 * np.pi / len(th)


def cpl(e, wt="mid"):
    W = np.sqrt(R**2 - 2 * R * e * np.cos(th) + e**2) - r
    assert (W / 2).min() >= A.min() and (W / 2).max() <= A.max()
    L = {"mid": R - W / 2, "outer": R + 0 * W, "inner": R - W}[wt]
    return float(np.sum(2 * sp(W / 2) * L) * dth)


c0 = cpl(0.0)
for e in (1.0, 2.0, 3.0):
    print(
        f"e {e}: Delta_mid {cpl(e) - c0:+.6f}  [outer {cpl(e,'outer') - cpl(0,'outer'):+.6f}, inner {cpl(e,'inner') - cpl(0,'inner'):+.6f}]"
    )


def half(p, e):
    return 0.5 * (-e * math.cos(p) + math.sqrt(R * R - (e * math.sin(p)) ** 2) - r)


e = 3.0
for b, lo, hi in [(4.1125, 4.100, 4.125), (5.1425, 5.140, 5.145), (6.1425, 6.130, 6.155)]:
    f = lambda x: math.degrees(brentq(lambda p: half(p, e) - x, 0, math.pi))
    print(
        f"e 3: birth {b} -> phi {f(b):.2f}; band (half-width +-0.05) {f(b - 0.05):.2f} to {f(b + 0.05):.2f}"
    )
