"""Q487: FROZEN forward predictions of the least-favourable-prior transitions for 18.7 < m <= 30 (bounded normal mean,
squared error), written before q486 reaches that range. Three models, each fitted ONLY on transitions m_t >= 8 up to and
including 18.678365 (the 27 -> 28 split bisected before q486 stopped):
  LAW   the O(rho^-2) local law of LFP_DERIVATION.md (rho^2 rho' = s0 [1 + (5/(8 pi^2))/rho^2 + 2 rho gamma],
        gamma from cos^2(pi theta/(2(m + 3.34)))), integrated from the m = 18 state's 4th-gap midpoint, one constant T;
  MLNM  K = c m ln m + b (Zhang's A ln A form), with K = K_before + 1/2 at each transition;
  FREEP K = c m^p + b.
PRE-REGISTERED: when q486 has located the transitions in (18.7, 30], the LAW passes if its max |error| in m is <= 0.04
and MLNM's max |error| exceeds twice the LAW's; FAIL for the law if its max |error| exceeds 0.08. Written to
lfp_forward_predictions.json with a timestamp.
v2 (23 Sep 07:58): an IMPORT of this module at 07:34:55 re-ran it, refitted on transitions inside the test range and
overwrote the frozen file. The fit window is now fixed to m_t <= 18.678365, the module writes only when FREEZE_OUT names
a file that does not exist, and the pre-registered values were restored and checked against the 04:47 printed table.
"""

import sys, json, time, numpy as np
from scipy.optimize import brentq, curve_fit

sys.path.insert(0, r"lfp_scripts")
_a = sys.argv
sys.argv = [sys.argv[0]]
import q484_lfp_bnm as Q

sys.argv = _a
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


d = json.load(open("results_q486_lfp_sym.json"))
tr = [tuple(t) for t in d["transitions"]]
row = min(d["rows"], key=lambda r: abs(r["m"] - 18.0))
m18 = row["m"]
J = row["J"]
v = np.array(row["v"])
pos = np.concatenate([v[: J - 1], [m18]])
allp = np.sort(np.concatenate([-pos, pos] + ([np.array([0.0])] if row["centre"] else [])))
dist = np.sort(m18 - allp[allp >= -1e-12])
gaps = np.diff(dist)
d0 = dist[3] + gaps[3] / 2
rho0 = 1 / gaps[3]
mt = np.array([t[0] for t in tr] + [18.678365])
K0 = np.array([t[1] for t in tr] + [27])
_u = np.unique(np.round(mt, 6), return_index=True)[1]
mt, K0 = mt[_u], K0[_u]  # v2: 18.678365 once only
sel = (mt >= 8) & (
    mt <= 18.678365 + 1e-9
)  # v2 (08:05): the fit window is FIXED to the pre-registered transitions
NF = lambda mm: 2 * (3.5 + profile_N(mm, d0, rho0))
Ts = np.array([NF(x) - k for x, k in zip(mt[sel], K0[sel])])
T = Ts.mean()
Kh = K0 + 0.5
(c1, b1), _ = curve_fit(lambda mm, c, b: c * mm * np.log(mm) + b, mt[sel], Kh[sel])
(c2, p2, b2), _ = curve_fit(
    lambda mm, c, p, b: c * mm**p + b, mt[sel], Kh[sel], p0=(0.7, 1.2, 2.5), maxfev=20000
)
print(
    f"fit on {sel.sum()} transitions (8 <= m_t <= 18.678365): start d0 = {d0:.4f}, rho0 = {rho0:.5f}; LAW T = {T:.4f} (spread {np.ptp(Ts):.4f}); "
    f"MLNM c = {c1:.5f}, b = {b1:+.4f}; FREEP c = {c2:.5f}, p = {p2:.4f}, b = {b2:+.4f}"
)
pred = []
for K in range(28, 60):
    try:
        mL = brentq(lambda mm: NF(mm) - T - K, 15, 40, xtol=1e-7)
    except ValueError:
        break
    if mL > 30:
        break
    mM = brentq(lambda mm: c1 * mm * np.log(mm) + b1 - (K + 0.5), 5, 60)
    mP = brentq(lambda mm: c2 * mm**p2 + b2 - (K + 0.5), 5, 60)
    kind = "split" if K % 2 == 1 else "appear"
    pred.append(dict(K_before=K, kind=kind, LAW=mL, MLNM=mM, FREEP=mP))
    print(
        f"   {K} -> {K + 1} ({kind:6s}): LAW {mL:8.4f}   MLNM {mM:8.4f}   FREEP {mP:8.4f}   MLNM - LAW {mM - mL:+.4f}"
    )
import os

OUT = os.environ.get("FREEZE_OUT", "")
if OUT:  # v2: writes only to an explicitly named NEW file, never over one
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    json.dump(
        dict(
            written=time.strftime("%Y-%m-%d %H:%M:%S"),
            fit=dict(T=T, d0=d0, rho0=rho0, mlnm=[c1, b1], freep=[c2, p2, b2]),
            predictions=pred,
        ),
        open(OUT, "w"),
        indent=1,
    )
    print(f"written to {OUT}")
