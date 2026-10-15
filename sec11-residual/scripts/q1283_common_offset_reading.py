"""q1283 (8 Oct 2026, EXPLORATORY reading, no item, no verdict): is the slow rise of c_k = R_k / R_NPMLE (equal v^2) with K_b, and its jump at K_b 50, produced by ONE additive offset delta common to all series, R_obs = R_true - delta (for example a small common bias of the median-sum law)? For delta on a grid the residuals R_k + delta are re-formed, c_k(K_b) recomputed at the v^2 of each k event (NPMLE interpolated in ln(R + delta)), and the flatness of c_k over K_b 29 to 48 (K_b 38 excluded) is measured by the sum over k = 2, 3, 4, 5 of the variance of ln c_k.
The data and the inclusion rule are those of q1282 (imported as text through its code); the grid, the statistic and the use of K_b 29 to 48 are fixed here before any delta is evaluated.
Control (first): with synthetic R_k = c_k U(v^2) - delta0 (delta0 = 0.007e-9, c_k = 1.8, 2.55, 3.3, 4.0, U = the measured NPMLE R_NP + delta0) the grid minimum is at delta0 within the grid step.
Usage: py q1283_common_offset_reading.py  (from temporal_within/centre_scripts/)"""

import os, sys, io, contextlib, json
import numpy as np

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1282_extended_decay_reading as D
data = D.data
dn = np.array(data[1])
o = np.argsort(dn[:, 2])


def stat(delta, dat):
    xs = dn[o, 2]
    ys = np.log(dn[o, 1] + delta)
    tot = 0.0
    out = {}
    for k in (2, 3, 4, 5):
        c = [
            (K, np.log((r + delta) / np.exp(np.interp(v, xs, ys))))
            for K, r, v in dat[k]
            if 29 <= K <= 48 and K != 38 and xs.min() <= v <= xs.max()
        ]
        lc = np.array([x[1] for x in c])
        tot += float(np.var(lc))
        out[k] = (np.array([x[0] for x in c]), lc)
    return tot, out


grid = np.arange(-0.004, 0.0201, 0.001)
# control on synthetic data
U = dn[o, 1] + 0.007
syn = {1: data[1]}
for k, c0 in zip((2, 3, 4, 5), (1.8, 2.55, 3.3, 4.0)):
    syn[k] = [(K, c0 * float(np.exp(np.interp(v, dn[o, 2], np.log(U)))) - 0.007, v) for K, r, v in data[k]]
gs = [stat(d, syn)[0] for d in grid]
ok = abs(grid[int(np.argmin(gs))] - 0.007) <= 0.0011
print(f"control: synthetic minimum at delta {grid[int(np.argmin(gs))]:.3f} (true 0.007): {ok}")
if not ok:
    print("READING q1283: VOID (control)")
    sys.exit(2)
vals = [(d, stat(d, data)[0]) for d in grid]
best = min(vals, key=lambda t: t[1])
print("delta (1e-9) : sum of var(ln c_k), k = 2..5  " + "; ".join(f"{d:.3f}: {v:.2e}" for d, v in vals))
fine = np.arange(best[0] - 0.002, best[0] + 0.0021, 0.0005)
vf = [(d, stat(d, data)[0]) for d in fine]
b2 = min(vf, key=lambda t: t[1])
print(
    f"minimum on the grid at delta = {best[0]:.3f}e-9 (sum of variances {best[1]:.2e}); refined {b2[0]:.4f}e-9 ({b2[1]:.2e}); at delta = 0: {dict(vals)[0.0] if 0.0 in dict(vals) else vals[4][1]:.2e}"
)
tot, out = stat(b2[0], data)
tot0, out0 = stat(0.0, data)
for k in (2, 3, 4, 5):
    K, lc = out[k]
    K0, lc0 = out0[k]
    sl = np.polyfit(K, lc, 1)[0]
    sl0 = np.polyfit(K0, lc0, 1)[0]
    print(
        f"k = {k}: events {len(K)}; c at K_b {K[0]}..{K[-1]}: {np.exp(lc[0]):.4f} .. {np.exp(lc[-1]):.4f} (delta = {b2[0]:.4f}), {np.exp(lc0[0]):.4f} .. {np.exp(lc0[-1]):.4f} (delta = 0); slope of ln c per K_b {sl:+.5f} (delta) against {sl0:+.5f} (0)"
    )
print("READING q1283 (exploratory, no verdict)")
