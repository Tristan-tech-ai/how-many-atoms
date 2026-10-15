"""q1396 reading (9 Oct 2026, EXPLORATORY, no verdict; py q1396_edge_depth_constancy.py from temporal_within/small_gap_q206/): constancy of the depths s_j = A - u_(m+1-j) of the outermost interior atoms below the wall over all stored OPTIMAL capacity states with K >= 60 (40-digit files under ..): min, max and spread for j = 1..8, separately for even and odd K, and the dependence on K (median per K for K = 60, 70, 80). Expected before the run: s_1 to s_3 constant to 1e-4 relative (as printed by q1395 to 5 digits), drifting inward faster for larger j; spread grows with j. Control: the spread of a synthetic family with s_j fixed and noise 1e-12 is below 1e-10 and a synthetic drift of 1e-6 per atom index is detected."""

import glob, json, os, re
import numpy as np
from mpmath import mp, mpf

mp.dps = 40
st = {}
for f in glob.glob(os.path.join("..", "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
    if "eps" in f:
        continue
    try:
        j = json.load(open(f))
        K = int(j["K"])
        res = float(j["residual"])
    except Exception:
        continue
    if K < 60 or j.get("verdict", "OPTIMAL") != "OPTIMAL" or res > 1e-20:
        continue
    A = mpf(j["A"])
    u = [mpf(x) for x in j["u"]]
    st[(K, round(float(A), 6))] = [float(A - u[-1 - i]) for i in range(8)]
keys = sorted(st)
S = np.array([st[k] for k in keys])
Ks = np.array([k[0] for k in keys])
# control
rng = np.random.default_rng(1)
base = np.array([2.7, 4.9, 6.8, 8.7, 10.5, 12.2, 13.9, 15.5])
syn = base + rng.normal(0, 1e-12, (50, 8))
ctl1 = (syn.max(0) - syn.min(0)).max() < 1e-10
syn2 = base + 1e-6 * np.arange(50)[:, None] * np.arange(8)[None, :]
ctl2 = (syn2.max(0) - syn2.min(0))[7] > 1e-5
print(
    "states",
    len(keys),
    "K from",
    Ks.min(),
    "to",
    Ks.max(),
    "; control constant family:",
    "OK" if ctl1 else "FAILED",
    "; control drift detected:",
    "OK" if ctl2 else "FAILED",
)
for par, mask in (("even", Ks % 2 == 0), ("odd", Ks % 2 == 1)):
    X = S[mask]
    print(par, "states", int(mask.sum()))
    for j in range(8):
        print(
            f"   s_{j+1}: min {X[:, j].min():.8f}, max {X[:, j].max():.8f}, spread {X[:, j].max() - X[:, j].min():.3e}, spread/value {(X[:, j].max() - X[:, j].min()) / X[:, j].mean():.2e}"
        )
for K in (60, 70, 80):
    m = Ks == K
    if m.any():
        print(
            f"K {K}: {int(m.sum())} states, median s_1..s_4:",
            [f"{np.median(S[m][:, j]):.8f}" for j in range(4)],
            "; s_1 min/max",
            f"{S[m][:, 0].min():.8f} {S[m][:, 0].max():.8f}",
        )
print("READING q1396 done")
