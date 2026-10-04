"""Capacity on the eccentric annulus R 15, r 10: Delta(e) = int eps(W/2) (R - W/2) d theta - 2 pi R_mid eps(2.5), in e^(C + log 2 pi e)."""

import json, math
import numpy as np
from scipy.interpolate import CubicSpline

d = json.load(open(r"scratch\eps_cap_scan.json"))
pts = sorted((float(a), v["eps"]) for a, v in d.items() if v and v["acc"])
A = np.array([p[0] for p in pts])
Yv = np.array([p[1] for p in pts])
sp = CubicSpline(A, Yv)
R, r = 15.0, 10.0
th = np.linspace(0, 2 * np.pi, 20001)[:-1]
dth = 2 * np.pi / len(th)


def cpl(e, wt="mid"):
    W = np.sqrt(R**2 - 2 * R * e * np.cos(th) + e**2) - r
    a = W / 2
    assert a.min() >= A.min() and a.max() <= A.max()
    L = {"mid": R - W / 2, "outer": R + 0 * W, "inner": R - W}[wt]
    return float(np.sum(sp(a) * L) * dth)


c0 = cpl(0.0)
h2 = math.log(2 * math.pi * math.e)
print(f"concentric coupling {c0:+.6f} (2 pi 12.5 eps(2.5) = {2*math.pi*12.5*float(sp(2.5)):+.6f})")
for e in (0.5, 1.0):
    dm = cpl(e) - c0
    print(
        f"e {e}: Delta_mid {dm:+.6f} in e^(C+h)   [outer {cpl(e,'outer') - cpl(0,'outer'):+.6f}, inner {cpl(e,'inner') - cpl(0,'inner'):+.6f}]"
    )
