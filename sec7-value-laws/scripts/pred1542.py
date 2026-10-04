"""Local-width prediction for the eccentric annulus (R 18, r 8): Delta(e) = oint 2 delta0(W/2) ds_mid - 2 pi R_mid 2 delta0(5)."""

import json, math
import numpy as np
from scipy.interpolate import CubicSpline

d = json.load(open(r"scratch\delta0_scan_q858.json"))
pts = sorted((float(a), v["delta0"]) for a, v in d.items() if v["acc"])
A = np.array([p[0] for p in pts])
Y = np.array([p[1] for p in pts])
sp = CubicSpline(A, Y)
print("accepted scan points", len(A), "range", A.min(), A.max())
R, r = 18.0, 8.0
th = np.linspace(0, 2 * np.pi, 20001)[:-1]
dth = 2 * np.pi / len(th)


def coupling(e, ds="mid"):
    W = (
        np.sqrt(R**2 - 2 * R * e * np.cos(th) + e**2) - r
    )  # outer point to the inner circle, along the line to its centre
    a = W / 2
    if a.min() < A.min() or a.max() > A.max():
        raise ValueError("outside scan")
    if ds == "mid":
        L = R - W / 2
    elif ds == "outer":
        L = R + 0 * W
    else:
        L = R - W
    return float(np.sum(2 * sp(a) * L) * dth), float(a.min()), float(a.max())


c0 = coupling(0.0)[0]
print(f"concentric coupling 2 pi R_mid 2 delta0(5) = {c0:+.6f}  (R_mid 13, delta0(5) {float(sp(5.0)):+.4e})")
for e in (1.0, 2.0):
    cm, amin, amax = coupling(e)
    co = coupling(e, "outer")[0]
    ci = coupling(e, "inner")[0]
    print(
        f"e {e}: a in [{amin:.3f}, {amax:.3f}]  coupling(mid) {cm:+.6f}  Delta(mid) {cm - c0:+.6f}   [ds variants: outer {co - coupling(0.0,'outer')[0]:+.6f}, inner {ci - coupling(0.0,'inner')[0]:+.6f}]"
    )
