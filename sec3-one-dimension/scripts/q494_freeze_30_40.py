"""Q494: FROZEN forward predictions of the LFP transitions in (30, 40], written BEFORE q492 reaches m = 30, from the
04:47:51 frozen fit read from lfp_forward_predictions.json (T, d0, rho0, MLNM (c, b), FREEP (c, p, b); nothing refitted).
profile_N is copied from q487 lines 23-31 as text (q487 is not imported, amendment 634). Writes
lfp_forward_predictions_30_40.json once and refuses to overwrite it."""

import json, os, time, numpy as np
from scipy.optimize import brentq

pi = np.pi
s0 = 1 / (16 * pi**2)


def profile_N(m, d0, rho0, c=5 / (8 * pi**2), dl=3.34, h=0.05):
    L = m + dl
    f = lambda d, y: np.array(
        [s0 * (1 + c / y[0] ** 2 + 2 * y[0] * (pi / L) / np.tan(pi * (d + dl) / (2 * L))) / y[0] ** 2, y[0]]
    )
    d = d0
    y = np.array([rho0, 0.0])
    while d < m - 1e-12:
        hh = min(h, m - d)
        k1 = f(d, y)
        k2 = f(d + hh / 2, y + hh * k1 / 2)
        k3 = f(d + hh / 2, y + hh * k2 / 2)
        k4 = f(d + hh, y + hh * k3)
        y = y + hh * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        d += hh
    return y[1]


F = json.load(open("lfp_forward_predictions.json"))
assert F["written"] == "2026-09-23 04:47:51"
p = F["fit"]
T, d0, rho0 = p["T"], p["d0"], p["rho0"]
(c1, b1), (c2, p2, b2) = p["mlnm"], p["freep"]
NF = lambda mm: 2 * (3.5 + profile_N(mm, d0, rho0))
# consistency: the 47 -> 48 LAW value must equal the frozen file's
chk = brentq(lambda mm: NF(mm) - T - 47, 15, 45, xtol=1e-7)
assert abs(chk - [q for q in F["predictions"] if q["K_before"] == 47][0]["LAW"]) < 1e-6
pred = []
for K in range(48, 90):
    mL = brentq(lambda mm: NF(mm) - T - K, 15, 60, xtol=1e-7)
    if mL > 40:
        break
    mM = brentq(lambda mm: c1 * mm * np.log(mm) + b1 - (K + 0.5), 5, 80)
    mP = brentq(lambda mm: c2 * mm**p2 + b2 - (K + 0.5), 5, 80)
    kind = "split" if K % 2 == 1 else "appear"
    pred.append(dict(K_before=K, kind=kind, LAW=mL, MLNM=mM, FREEP=mP))
    print(f"   {K} -> {K + 1} ({kind:6s}): LAW {mL:8.4f}   MLNM {mM:8.4f}   FREEP {mP:8.4f}")
OUT = "lfp_forward_predictions_30_40.json"
assert not os.path.exists(OUT), "refusing to overwrite"
json.dump(
    dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        fit_from="lfp_forward_predictions.json (04:47:51), unchanged",
        predictions=pred,
    ),
    open(OUT, "x"),
    indent=1,
)
print(f"written {OUT}, {len(pred)} transitions")
