"""q1564 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 28, PROOF_TRACKS A step 40): the edge pair as a limit in the law order, with the exact coefficients f18 and f20 of q1557 (ORD = 18 and 20).
q1557 (ORD = 18, 20): f18 = -54007/(293601280 pi^10) - 53549/(165150720 pi^12) - 215/(8388608 pi^16) + 1/(134217728 pi^18) + 71767/(83886080 pi^14) = -2.2215440e-09; f20 = -64764727/(190253629440 pi^12) - 395711/(13589544960 pi^10) - 11659519/(31708938240 pi^14) - 2963/(536870912 pi^18) + 1/(1073741824 pi^20) + 848207/(2013265920 pi^16) = -7.1487862e-10. The ratios f18/f16 = 0.349 and f20/f18 = 0.322 fall below the earlier 0.44 and 0.48.
Reading 72/73: the pair of the shallow window 12..22 of the dump K = 130 converges in the law order n_max (law through c10, c12, ..., so that n_max = 5 is c10, 8 is c16, 10 is c20) with the ratio 0.48 and the Aitken limit over n_max = 6, 7, 8 reproduces the 48 measured births K_b 83..130 to 4.9e-08 (spread 7e-09).
Predictions written before the run:
 P1 the Aitken limit over n_max = 8, 9, 10 on the window 12..22 reproduces the 48 measured births with largest |predicted - measured| below 2e-08;
 P2 the Aitken limits over (6, 7, 8), (7, 8, 9) and (8, 9, 10) converge: the nu0 difference between the successive limits falls by a factor of at least 3;
 P3 the rms in N on the window 12..22 keeps falling and is below 1e-08 at n_max = 10;
 P4 the limit over (8, 9, 10) lies within 3e-08 of the q1562 instrument pair (-0.093773743, +0.001160038) in each number.
Usage: py q1564_orders_9_10.py (from small_gap_q206/, about 40 s; executes q1558 silently for its functions).
"""

import io, contextlib, os, json
import numpy as np
import mpmath as mp

H = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(H, "q1558_series_beyond_c10.py"), "__name__": "q1558ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(H, "q1558_series_beyond_c10.py"), encoding="utf-8").read(), ns)
LAW, fit_law, ladder, Ks, meas, dder, brentq, laws = (
    ns["LAW"],
    ns["fit_law"],
    ns["ladder"],
    ns["Ks"],
    ns["meas"],
    ns["dder"],
    ns["brentq"],
    ns["laws"],
)
pim = mp.pi
f18 = (
    -mp.mpf(54007) / (293601280 * pim**10)
    - mp.mpf(53549) / (165150720 * pim**12)
    - mp.mpf(215) / (8388608 * pim**16)
    + 1 / (134217728 * pim**18)
    + mp.mpf(71767) / (83886080 * pim**14)
)
f20 = (
    -mp.mpf(64764727) / (190253629440 * pim**12)
    - mp.mpf(395711) / (13589544960 * pim**10)
    - mp.mpf(11659519) / (31708938240 * pim**14)
    - mp.mpf(2963) / (536870912 * pim**18)
    + 1 / (1073741824 * pim**20)
    + mp.mpf(848207) / (2013265920 * pim**16)
)
ctl = max(abs(float(f18) / -2.221544039357757e-09 - 1), abs(float(f20) / -7.148786185885020e-10 - 1))
print(
    f"control: exact expressions of f18, f20 against the decimals printed by q1557: largest relative difference {ctl:.1e}"
)
F = ns["ns28"]["F"]
F.extend([float(f18), float(f20)])
print(f"ratios f18/f16 {F[9] / F[8]:.3f}, f20/f18 {F[10] / F[9]:.3f}; coefficient list length {len(F)}")
for n in (9, 10):
    LAW[n] = laws(n)
d130 = ladder(130)
w = (12, 22)
pr = {}
for n in range(5, 11):
    dfn, Nfn, dNfn, _ = LAW[n]
    pr[n] = fit_law(d130, *w, dfn, Nfn, dNfn)
    print(f"n_max = {n}: window 12..22 pair ({pr[n][0]:+.9f}, {pr[n][1]:+.9f}) rms in N {pr[n][2]:.2e}")


def aitken(i, x):
    x6, x7, x8 = x
    d1, d2 = x7 - x6, x8 - x7
    return x8 - d2 * d2 / (d2 - d1), d2 / d1


lim = {}
for tri in ((6, 7, 8), (7, 8, 9), (8, 9, 10)):
    nu, rn = aitken(0, [pr[n][0] for n in tri])
    d0, rd = aitken(1, [pr[n][1] for n in tri])
    lim[tri] = (nu, d0)
    print(f"Aitken over n_max = {tri}: nu0 {nu:+.9f} (ratio {rn:.3f}), d0 {d0:+.9f} (ratio {rd:.3f})")


def offs(nu_, d0_, Hs=1e-3):
    dfn, Nfn, _, _ = LAW[10]
    rho = lambda a: brentq(lambda r: dfn(r) + d0_ - a, 0.05, 6.0, xtol=1e-15)
    Nn = lambda a: Nfn(rho(a)) + 0.5 - nu_
    ap = lambda K: brentq(
        lambda a: 2 * Nn(a) - K - 1 / 6 - dder((Nn(a + Hs) - Nn(a - Hs)) / (2 * Hs), 4), 30, 85, xtol=1e-14
    )
    return np.array([ap(K) - meas[K] for K in Ks])


mx = {}
for tri in lim:
    r = offs(*lim[tri])
    mx[tri] = np.abs(r).max()
    print(
        f"births law (n_max = 10 at the centre) with the Aitken limit over {tri}: mean offset {r.mean():+.3e}, spread {r.max() - r.min():.2e}, largest |.| {np.abs(r).max():.3e}"
    )
INS = (-0.093773743, 0.001160038)
a = [lim[(6, 7, 8)][0], lim[(7, 8, 9)][0], lim[(8, 9, 10)][0]]
d_a, d_b = abs(a[1] - a[0]), abs(a[2] - a[1])
p1 = mx[(8, 9, 10)] < 2e-8
p2 = d_b <= d_a / 3
p3 = pr[10][2] < 1e-8
p4 = abs(lim[(8, 9, 10)][0] - INS[0]) < 3e-8 and abs(lim[(8, 9, 10)][1] - INS[1]) < 3e-8
print(
    f"successive limits differ by {d_a:.2e} then {d_b:.2e} in nu0; limit over (8, 9, 10) minus q1562 pair: ({lim[(8, 9, 10)][0] - INS[0]:+.2e}, {lim[(8, 9, 10)][1] - INS[1]:+.2e})"
)
print(
    f"P1 largest miss below 2e-8: {'holds' if p1 else 'does not hold'}; P2 successive Aitken limits converge by a factor of 3: {'holds' if p2 else 'does not hold'}; P3 rms below 1e-8 at n_max = 10: {'holds' if p3 else 'does not hold'}; P4 limit within 3e-8 of the q1562 pair: {'holds' if p4 else 'does not hold'}"
)
print("READING q1564 done (labels by the script; no verdict word)")
